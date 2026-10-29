# Sherman QC shop-floor runbook

Audience: the person who installs and runs the production line PC. Package:
`deploy/shopfloor/` (see its README for the file layout). Cell contract:
[zeiss-cell-integration-spec.md](zeiss-cell-integration-spec.md). Plan of record:
`docs/production/PRODUCTION_LINE_PLAN.md`.

The line is **operator assist**. A green ALL_MEASURED_IN_TOLERANCE means "every required
characteristic was measured and passed with its uncertainty". It is not an automatic release
of the part. Grey "Undetermined" (UNKNOWN) and amber REVIEW always need a person. The line
never turns an unseen value into PASS or FAIL.

Paths below are the Linux defaults (`/opt/sherman-qc`, `/var/lib/sherman-qc`,
`/etc/sherman-qc/env`). On Windows, read them as `C:\ShermanQC`, `C:\ProgramData\ShermanQC\line`
and `C:\ProgramData\ShermanQC\env`. Commands that run Python use the app copy:

```bash
APP=/opt/sherman-qc/app; PY=/opt/sherman-qc/venv/bin/python
run() { sudo systemd-run --quiet --pipe --wait --collect --uid=shermanqc --gid=shermanqc \
          -p EnvironmentFile=/etc/sherman-qc/env -p WorkingDirectory="$APP" \
          --setenv=PYTHONPATH="$APP:$APP/backend" --setenv=PYTHONDONTWRITEBYTECODE=1 \
          "$PY" "$@"; }
```

`run` starts a one-off transient unit as `shermanqc` with the same `EnvironmentFile=` as the
services:
- systemd (PID 1) reads `/etc/sherman-qc/env` itself and parses it exactly as for the services, so
  JSON values such as `SHERMAN_LINE_STATIONS` arrive unchanged.
- No value of the file ever appears on a command line, so `ps` and `/proc/*/cmdline` cannot show
  `SECRET_KEY`.
- `--pipe` passes stdin and stdout through (the password pipe in section 2 works), and `--wait`
  returns the command's exit code.
- Never load the env file with `env $(cat ...)`, `xargs` or `. env` instead. The first two put every
  secret on argv; all three mangle quoted or JSON values.

The backup and retention tools need nothing from the env file except
`SHERMAN_LINE_LOG_RETENTION_DAYS` (retention's default `--log-days`).

---

## 1. Install

1. Prepare a **release source tree**: a clean checkout or tarball of the repo, without `data/`,
   `output/`, `outputs/`, `artifacts/`, `uploads/` or `live_scans/`. Those are development custody and
   never go to the shop floor. For an offline PC, also prepare a wheelhouse on a matching
   OS/Python. If the PC has no node/npm, add a prebuilt `frontend/react` dist.
2. Run `sudo deploy/shopfloor/linux/install.sh --source <tree> [--wheelhouse W] [--frontend-dist D]`.
   It does the following:
   - creates the `shermanqc` system user, `/var/lib/sherman-qc` (0700), `/var/backups/sherman-qc`
     (0700) and `/etc/sherman-qc/env` (0600, from `env.example`, with a generated `SECRET_KEY`);
   - creates the venv (offline with `--wheelhouse`);
   - builds the app copy (`engine_bundle build --profile app`, which builds the frontend into the copy);
   - builds the engine bundle from that copy (`--profile engine`, installed as
     `/opt/sherman-qc/engine/<bundle_id>`, and verifies it);
   - installs the systemd units, offers the admin bootstrap and enables the services.
3. Edit `/etc/sherman-qc/env`: set `SHERMAN_LINE_STATION_ID` (`[a-z0-9-]{1,32}`, e.g. `cell-1`) and
   `SHERMAN_LINE_INBOX`. Every `CHANGE_ME` must go. Use `SHERMAN_LINE_STATIONS` (JSON) for
   several cells.
4. Mount the ZEISS share (section 5), then `sudo systemctl start sherman-qc-line`.
5. Check:
   ```bash
   systemctl status sherman-qc-api sherman-qc-line
   curl -s http://127.0.0.1:9000/api/line/health/ready        # {"ready": true, "checks": {...}}
   ```

Windows: `deploy\shopfloor\windows\install-services.ps1` does the same with NSSM services.
- The service account (`-ServiceAccount`) must be able to read and write the ZEISS share.
  `LocalSystem` cannot reach SMB shares.
- The inbox is a UNC path (`\\zeiss-pc\export\cell-1`).
- Each service runs the venv python with `C:\ShermanQC\bin\service_launch.py`, which reads
  `C:\ProgramData\ShermanQC\env` **at every service start**. After editing the env file, restart
  the service (`nssm restart SHERMAN-QC-API`, then `nssm start` or `nssm restart SHERMAN-QC-LINE`).
  No re-install is needed, and nothing from the env file is stored in the registry.
- `C:\ShermanQC` (venv, releases, engine bundles, `bin\`) is writable only by Administrators and
  SYSTEM. The service account and Users can only read and execute it. The data root, backups and
  each release's `data\`, `uploads\`, `output\` are restricted to Administrators, SYSTEM and the
  service account.
- WinSW works as well: use the same command (`python service_launch.py <env> <app> -m <module> ...`),
  working directory, restart-on-exit and log directory.

## 2. First admin

Opening a new users DB seeds `admin` / `admin123`. With `SHERMAN_PRODUCTION_HARDENING=1` the API
**refuses to start** while that default password is active (DEFAULT_ADMIN_PASSWORD_ACTIVE).
Bootstrap a real admin:

```bash
printf '%s\n' "$PASSWORD" | run -m apps.production_line.security.bootstrap_admin \
    --users-db /opt/sherman-qc/app/data/users.db --username alice --password-stdin [--rotate]
```

- The password needs at least 12 characters and is never echoed.
- If the username is not `admin`, the seeded admin is deactivated.
- Exit codes: 0 done, 2 invalid input, 3 the user exists (pass `--rotate`), 4 auth backend failure.

## 3. Roles

| Role | May |
|---|---|
| viewer | read the station feed, jobs and results (`line.read`) |
| operator | also retry and cancel jobs, assign or dismiss quarantined scans (`line.operate`), edit parts (`parts.write`) |
| admin | everything, plus device tokens (`devices.manage`), metrics, deleting and importing parts |

Recipe release (`recipe.release`) goes to the roles in `SHERMAN_RELEASE_ROLES` (default `admin`;
`admin,operator` also lets operators release). Change it in the env file and restart the API.
Self-registration stays off (`SHERMAN_ALLOW_SELF_REGISTER=0`).

## 4. Station device tokens

A kiosk does not log in as a user. It holds a **device token** (`sdt_...`) that may only read its
own station's feed (`line.read`).

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:9000/api/auth/login -H 'Content-Type: application/json' \
          -d '{"username":"alice","password":"..."}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
curl -s -X POST http://127.0.0.1:9000/api/line/devices -H "Authorization: Bearer $TOKEN" \
     -H 'Content-Type: application/json' -d '{"station_id":"cell-1","label":"press brake kiosk"}'
# -> {"token_id":"dt_...","token":"sdt_...","shown_once":true}
curl -s http://127.0.0.1:9000/api/line/devices -H "Authorization: Bearer $TOKEN"            # list
curl -s -X DELETE http://127.0.0.1:9000/api/line/devices/dt_... -H "Authorization: Bearer $TOKEN"   # revoke
```

- The token is shown **once** and only its sha256 is stored. Type it into the station screen's
  token prompt; the browser keeps it in that kiosk profile's localStorage.
- A revoked token gets 401 at the next request, and its SSE stream closes at the next heartbeat.
- Issue and revoke events are audited (DEVICE_TOKEN_ISSUED / DEVICE_TOKEN_REVOKED).

## 5. SMB mount of the ZEISS export share (Linux)

The inbox is polled, never watched: SMB change notifications are unreliable. Mount the cell's
export folder read-write, so accepted scans can be moved out of it:

```
# /etc/sherman-qc/smb-credentials (root:root 0600)
username=CHANGE_ME
password=CHANGE_ME
domain=CHANGE_ME

# /etc/fstab: one line per station
//zeiss-pc/ScanExport/cell-1  /mnt/zeiss/cell-1  cifs  credentials=/etc/sherman-qc/smb-credentials,uid=shermanqc,gid=shermanqc,file_mode=0600,dir_mode=0700,vers=3.0,actimeo=1,_netdev,x-systemd.automount,x-systemd.requires=network-online.target  0  0
```

- `actimeo=1` keeps the size and mtime the poller sees at most 1 s stale. The default attribute
  cache can make a file look stable while ZEISS is still writing it.
- `sherman-qc-line.service` has `RequiresMountsFor=/mnt/zeiss`; edit it if the mount point differs.
- The service refuses to start with STATION_INBOX_MISSING when the inbox directory does not
  exist. A missing mount is never silently replaced by an empty local folder.
- `mount -a && ls /mnt/zeiss/cell-1` to test; `sudo -u shermanqc touch /mnt/zeiss/cell-1/.probe && rm ...`
  checks write access.
- Accept rules: the size and mtime are stable for 3 polls, at least 2 s old, at least 1 KiB, and
  temp names are ignored (`*.tmp`, `*.part`, `*.partial`, `~`, dotfiles). With
  `"sidecar_mode":"required"`, a `<file>.done` sidecar must also exist. A truncated or invalid
  file is quarantined only after it stayed unchanged for 60 s.

## 6. Kiosk

- **Linux.** As the desktop user, autostart `/opt/sherman-qc/bin/kiosk-launch.sh cell-1`, e.g. with
  `~/.config/autostart/sherman-kiosk.desktop` and `Exec=/opt/sherman-qc/bin/kiosk-launch.sh cell-1`.
  - It opens Chromium `--kiosk --app=http://127.0.0.1:9000/station/cell-1` with the dedicated,
    persistent profile `~/.local/share/sherman-qc-kiosk/cell-1`.
  - Never incognito or guest: the device token lives in that profile.
  - Deleting the profile forces a new token prompt.
- **Windows.** `kiosk-launch.ps1 -StationId cell-1` opens Edge `--app=... --start-fullscreen` with a
  persistent profile under `%LOCALAPPDATA%\ShermanQC\kiosk\cell-1`.
  - Edge's own kiosk mode (`-EdgeKiosk`: `--kiosk --edge-kiosk-type=fullscreen`) runs InPrivate and
    **forgets the token at every restart**.
  - To lock the PC down, use Windows assigned access with Edge as the only app and "Normal browsing",
    or a shell launcher, rather than `-EdgeKiosk`.
- The screen is 16:9 (1920×1080 recommended), touch and mouse. It updates by itself (SSE with a
  polling fallback) within seconds of a job finishing.

## 7. Daily operation

Station screen states:
- **Green** means all required characteristics were measured and passed.
- **Red** OUT_OF_TOLERANCE_FOUND means at least one characteristic FAILs beyond its uncertainty.
- **Amber** REVIEW_REQUIRED means a value sits inside the guard band, a stage or order was
  questioned, or there are unplanned creases.
- **Blue** STAGE_IN_TOLERANCE means the part is not finished yet. It is never green.
- **Grey** INCOMPLETE / INVALID_INSPECTION mean something could not be measured. Rescan or check
  the capture; never treat these as a defect.

Operator actions (all `line.operate`; the UI offers them, the API is shown for scripting):

| Action | API | Notes |
|---|---|---|
| Quarantine list | `GET /api/line/quarantine` | Reasons: UNMATCHED_SKU, AMBIGUOUS_SKU, UNPARSEABLE_FILENAME, REVISION_MISMATCH, PART_NOT_READY (assignable); UNSUPPORTED_FORMAT, PLY_HEADER_INVALID, PLY_TRUNCATED, STL_INVALID, EMPTY_SCAN, SIDECAR_INVALID, READ_ERROR (re-export from the cell) |
| Assign a quarantined scan | `POST /api/line/quarantine/{scan_file_id}/assign {"sku": "..."}` | Uses the SKU's active released recipe. Refusals: 404 SKU_NOT_FOUND; 409 PART_NOT_READY (no compiled recipe); 409 RECIPE_NOT_ACTIVE / RECIPE_RETIRED (when a `recipe_id` is sent) |
| Dismiss | `POST /api/line/quarantine/{scan_file_id}/dismiss {"note": "..."}` | |
| Retry a failed job | `POST /api/line/jobs/{job_id}/retry` | new job, same scan; fails with SCAN_MISSING if retention pruned the scan |
| Cancel | `POST /api/line/jobs/{job_id}/cancel` | a running step is stopped (process-group kill) |
| Status | `GET /api/line/status` | queue, heartbeats, last scan, last job |

A wrong-SKU or wrong-revision export goes to quarantine. That is normal, not a fault.
Re-dropping identical bytes is idempotent: it gives SCAN_DUPLICATE and no second job.

## 8. Health checks

`GET /api/line/health/ready` needs no auth and answers 503 when not ready. Each check is `ok` or `fail`:

| Check | Fails when | Do |
|---|---|---|
| `configured` | line settings missing or invalid | `journalctl -u sherman-qc-api`; fix the env file |
| `db` | `db/line.sqlite3` missing or not writable | disk full or permissions; `ls -l /var/lib/sherman-qc/db` |
| `disk` | free space on the data root below the minimum | run retention (section 11) or add disk |
| `runner` | no runner heartbeat recently | `systemctl status sherman-qc-line`; `journalctl -u sherman-qc-line` |
| `hotfolder:<station>` | that station's poller is not heartbeating | SMB mount gone? (`ls /mnt/zeiss/<station>`); restart the line service |

`GET /api/line/metrics` (admin) gives Prometheus text. Logs: `journalctl -u sherman-qc-*` and
`/var/lib/sherman-qc/logs/` (JSON lines; credentials and SSE tokens are redacted).

## 9. Upgrade and rollback (by engine bundle id)

- **Upgrade.** Re-run `install.sh --source <new tree>`. It works in this order:
  1. builds `releases/app-<UTC>` and the engine bundle `engine/<bundle_id>`, and verifies both;
  2. stops the services;
  3. copies `data/`, `uploads/` and `output/` from the old app copy;
  4. switches `app` and `engine/current`;
  5. records the change in `releases/HISTORY` and starts the services.

  `bundle_id` is a content hash (first 16 hex of the sha256 over files, dependency versions and
  engine config), so identical code gives the identical id. Every job records the
  `engine_bundle_id` it ran with (`GET /api/line/jobs/{id}`), and so does every result
  (`engine.bundle_id`).
- **Roll back the engine only** (the app stays):
  ```bash
  ls /opt/sherman-qc/engine/                      # installed bundle ids
  sudo ln -sfn /opt/sherman-qc/engine/<old_id> /opt/sherman-qc/engine/current.new && \
    sudo mv -T /opt/sherman-qc/engine/current.new /opt/sherman-qc/engine/current
  sudo systemctl restart sherman-qc-line          # the service re-verifies the bundle byte for byte
  ```
  Recipes compiled by a newer E1 carry that compiler version. If the old engine refuses a
  compiled part, the job fails closed (COMPILED_PART_HASH_MISMATCH / COMPILE_BLOCKED) and is never
  measured wrongly. Recompile, or roll the app back too.
- **Roll back the app**: stop the services, point `app` at the previous `releases/app-*` (listed in
  `HISTORY`), and if needed point `engine/current` at the bundle listed on the same line. Then
  start the services.
  - The legacy state is copied, not moved, at each upgrade, so the previous release still holds
    its own `data/` as of the upgrade time. Copy `data/` forward if users were added since.
  - The line state (`/var/lib/sherman-qc`) is shared and stays put.
- `install.sh --keep-releases N` (default 3) removes older releases, never the active or the
  previous one. Engine bundles are never removed automatically.

## 10. Backup and restore

The daily `sherman-qc-backup.timer` runs:

```bash
python -m apps.production_line.ops.backup create --data-root /var/lib/sherman-qc --app-root /opt/sherman-qc/app \
    --out /var/backups/sherman-qc --include-job-results --keep 14
```

- The backup file is `sherman-qc-backup-<UTC>.tar.gz` (0600). It holds an SQLite **online backup**
  of every `db/*.sqlite3` and of the app's `data/users.db` and `data/qc_jobs.db`, which is safe
  while the line runs.
- It also holds `recipes/` (frozen CAD, drawings, flats, compiled parts), `config/`, and with
  `--include-job-results` each job's JSON results and PDF.
- A manifest records every member's sha256, each tree's per-file index and each DB's
  `integrity_check`.
- Not included: raw scans (`archive/scans`) and heavy job artifacts. Snapshot the data-root volume
  if those must survive a disk loss.
- `secrets/` is included only with `--include-secrets`. It holds the stream-token key, which is
  short-lived and regenerated if lost; device tokens survive without it, because only their hashes
  are stored in `security.sqlite3`.
- **Copy `/var/backups/sherman-qc` off the machine** (NAS or USB) on a schedule. A backup on the
  same disk is not a backup.
- **Not in the backup, keep separately:** `/etc/sherman-qc/env` (`SECRET_KEY`, roles, station
  config) and `/etc/sherman-qc/smb-credentials`. On Windows these are `C:\ProgramData\ShermanQC\env`
  and the service account's password. A restore onto a new disk or PC needs them.
  - Keep one copy off the machine, encrypted (for example
    `sudo tar -C /etc -cz sherman-qc | gpg -c > sherman-qc-etc.tar.gz.gpg`). Store it with the
    password manager's entry for the PC, not on the backup share in clear text.
  - Without the old `SECRET_KEY` every issued login token becomes invalid; users log in again.
    Device tokens are unaffected, because they are stored hashed in `security.sqlite3`.
- **Record each backup's sha256 off the machine.** `create` prints it (`"sha256"` in its JSON line;
  for the timer it is in `journalctl -u sherman-qc-backup`). The manifest inside an archive proves
  only that the archive is internally consistent, not who wrote it. Anyone who can write
  `/var/backups/sherman-qc` (root and `shermanqc`) could forge a consistent archive.
  `verify` / `restore --expect-sha256 <sha>` refuse any other file (HASH_MISMATCH).

Verify a backup any time with
`run -m apps.production_line.ops.backup verify --backup FILE [--expect-sha256 SHA]`. It checks every
hash and DB integrity and restores nothing.

Restore onto a fresh install or a new disk:

```bash
sudo systemctl stop sherman-qc-line sherman-qc-api
# 0. on a new PC: install first (section 1), then put the kept /etc/sherman-qc/env and
#    smb-credentials back (root:root 0600).
# 1. targets must be empty: a new data root, and an app copy whose data/ is absent or empty
sudo mv /var/lib/sherman-qc /var/lib/sherman-qc.broken && sudo mv /opt/sherman-qc/app/data /opt/sherman-qc/app/data.broken
# 2. shermanqc cannot create these itself (/var/lib and the app copy are root-owned): pre-create them empty
sudo install -d -o shermanqc -g shermanqc -m 0700 /var/lib/sherman-qc /opt/sherman-qc/app/data
# 3. the backup must be readable by shermanqc; a copy from USB goes into the backup dir first
sudo install -o shermanqc -g shermanqc -m 0600 /media/usb/sherman-qc-backup-<UTC>.tar.gz /var/backups/sherman-qc/
# 4. restore. Staging goes to the backup dir (on disk, owned by shermanqc); /tmp may be a small tmpfs.
run -m apps.production_line.ops.backup restore --backup /var/backups/sherman-qc/sherman-qc-backup-<UTC>.tar.gz \
    --data-root /var/lib/sherman-qc --app-root /opt/sherman-qc/app \
    --staging-dir /var/backups/sherman-qc --expect-sha256 <sha256 recorded at create time>
sudo systemctl start sherman-qc-api sherman-qc-line
```

The staging directory needs free space for the uncompressed backup. It is a private 0700
directory and is removed afterwards. Restore refuses, **before** writing anything:
- non-empty targets (TARGET_NOT_EMPTY), and targets it cannot create or write (TARGET_INVALID,
  "pre-create it");
- a file other than the recorded one (HASH_MISMATCH with `--expect-sha256`);
- any hash or integrity mismatch (HASH_MISMATCH, DB_INTEGRITY);
- a manifest whose files would land outside their tree (`recipes.tar.gz` may only hold `recipes/...`,
  and so on) or two entries that restore to the same path (BACKUP_INVALID). It restores the line DB's WAL journal
mode and recreates the data-root layout with 0700 directories. Afterwards, jobs whose raw scans
were not in the backup can be viewed (results, PDF) but not retried (SCAN_MISSING).

## 11. Retention

```bash
run -m apps.production_line.ops.retention --data-root /var/lib/sherman-qc --days 90            # dry run: prints the plan
run -m apps.production_line.ops.retention --data-root /var/lib/sherman-qc --days 90 --apply    # deletes
```

- The age of a job is its `finished_at` in the line DB, and only **terminal** jobs (completed,
  failed, cancelled) older than N days are touched.
- Pruned: the staged scan copy, the compiled stage clouds, E3's registered scan, deviations, labels
  and display meshes, and the viewer assets and `overview.png`.
- Kept: `job.json`, `recipe.json`, every `*.json` (results), every `*.log` and `report.pdf`.
  An old job therefore still shows its table and PDF, without the 3D view.
- Archived raw scans are pruned only when every job using them is terminal and old, and the scan
  is not quarantined (still assignable).
- Events older than `--log-days` (default `SHERMAN_LINE_LOG_RETENTION_DAYS`, 30) are deleted from the
  line DB. Event ids keep growing, so station cursors stay valid.
- Symlinks are never followed. Exit 3 means some deletions failed (listed in `errors`).

Choose N together with the backup policy. Pruned scans are gone unless the volume was
snapshotted.

## 12. Troubleshooting by code

**Startup refusals** (the service exits 2 and journald shows the code):

| Code | Meaning / fix |
|---|---|
| `LINE_DISABLED` | `SHERMAN_LINE_ENABLED` is not 1. The API serves the legacy app and line routes answer 503; the line service exits. |
| `DATA_ROOT_MISSING` / `DATA_ROOT_NOT_ABSOLUTE` / `DATA_ROOT_INSIDE_REPO` | fix `SHERMAN_LINE_DATA_ROOT`: absolute, outside the app copy and the source tree |
| `STATION_INVALID` | station id pattern, inbox not absolute or overlapping another inbox or the data root, unknown JSON keys |
| `VALUE_INVALID` | a malformed value, or the app copy lacks `source_repo_root` in its manifest (rebuild with install.sh) |
| `RSS_LIMIT_INVALID` / `FILENAME_REGEX_INVALID` | `SHERMAN_LINE_RSS_LIMIT_MB` outside 256..32768; the regex needs `(?P<sku>...)` |
| `MULTIPLE_WORKERS` | `WEB_CONCURRENCY` / `UVICORN_WORKERS` must be 1 |
| `SECRET_KEY_MISSING_OR_DEFAULT` / `PRODUCTION_FLAG_MISSING` / `DEFAULT_ADMIN_PASSWORD_ACTIVE` | hardening: set a 32+ character `SECRET_KEY` and `PRODUCTION=1`, and bootstrap an admin (section 2) |
| `ENGINE_BUNDLE_MISSING` / `ENGINE_BUNDLE_INVALID` | `engine/current` missing or modified. Reinstall, or point it at a verified bundle (section 9). |
| `STATION_INBOX_MISSING` | the SMB share is not mounted (section 5) |

**Job failure codes** (`GET /api/line/jobs/{id}` → `failure.code`). A failed job is never a
part FAIL.

| Code | Meaning / fix |
|---|---|
| `RECIPE_MISSING` / `RECIPE_SHA_MISMATCH` | the recipe the job pinned is gone or changed: do not edit `recipes/` by hand; retry after a release |
| `RECIPE_CAD_MISSING` / `RECIPE_CAD_HASH_MISMATCH` | frozen CAD missing or altered: restore `recipes/` from backup |
| `COMPILED_PART_MISSING` / `COMPILED_PART_HASH_MISMATCH` / `COMPILE_BLOCKED` | the compiled part is missing, altered, superseded or blocked. Re-release the recipe; check the onboarding page for compile blockers. |
| `SCAN_MISSING` / `SCAN_HASH_MISMATCH` | archived scan gone (retention) or altered; re-export from the cell |
| `WORKER_TIMEOUT` | the engine exceeded 180 s and was killed; not auto-retried. Check the export density (ZEISS spec section 6) and report it |
| `WORKER_RSS_LIMIT` | the engine exceeded 32 GiB (ENGINE_IO-A5); not auto-retried; usually an over-dense export |
| `WORKER_ERROR` / `WORKER_CRASH` / `WORKER_NO_RESULT` / `ENGINE_OUTPUT_INVALID` | engine defect: keep the job workspace, send `jobs/<id>/logs/attempt-*/engine_step.log` to engineering |
| `POSTPROCESS_TIMEOUT` / `POSTPROCESS_ERROR` | viewer assets or PDF failed; the engine result is kept in `jobs/<id>/engine/`; retry |
| `WORKER_INTERRUPTED` | the service was killed mid-job and a step could not be stopped; retry |
| `CANCELLED` | an operator cancelled |
| `INTERNAL_ERROR` | report with logs |

**Validity states** on a completed job (shown grey, never as a defect):
- `INSUFFICIENT_CAPTURE` / `TIMEOUT`: rescan with better coverage.
- `UNTRUSTED_REGISTRATION`: check the fixture and that the right part was scanned.
- `UNIT_MISMATCH_SUSPECTED`: the export is not in mm (ZEISS spec "units").
- `WRONG_PART_SUSPECTED` / `MIRRORED_PART_SUSPECTED`: wrong SKU or a mirror part. Check the label.
- `UNSUPPORTED_INPUT`: not a readable PLY or STL.

**Backup and retention**:
- backup: `OUT_FORBIDDEN`, `DATA_ROOT_INVALID`, `TARGET_NOT_EMPTY`, `TARGET_INVALID`, `HASH_MISMATCH`,
  `DB_INTEGRITY`, `BACKUP_INVALID`, `VALUE_INVALID`;
- retention: `NOT_A_DATA_ROOT`, `DATA_ROOT_INVALID`, `EVENTS_SCHEMA_UNKNOWN` (the line DB schema is
  newer than the tool; file pruning still runs, event pruning is skipped).

## 13. Engineering checks on the installed line

`python -m apps.production_line.ops.synthetic_scorecard --out /tmp/sc --cases 6 --engine-root /opt/sherman-qc/engine/current`
runs a frozen, seeded synthetic grid through a private copy of the line (store, runner, the
installed engine bundle and the render step). It writes `scorecard.md`, headed
**SYNTHETIC ENGINEERING CAPABILITY — NOT PHYSICAL ACCURACY — NOT PROMOTION EVIDENCE**.
- Use it after an install or upgrade as a smoke test of the whole software stack.
- It is never a statement about real-part accuracy. Physical qualification follows
  `docs/research/2026-10-17_production_line_accuracy_protocol.md`.
