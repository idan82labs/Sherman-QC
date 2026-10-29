# SLDPRT → STEP (AP214, mm)

The CAD engine needs a STEP B-rep for each part. Many parts exist only as SolidWorks `.SLDPRT` files, and
only SolidWorks can convert them. The conversion runs at Sherman, on any PC that already has SolidWorks.
The STEP files and the manifest come back to us. We validate and import them with:
`python -m apps.production_line.recipes.cad_import validate|import` (see below).

There are three ways to convert. All of them open the parts read-only and never save them.

| | What | Who | Manifest |
|---|---|---|---|
| **A (primary)** | VBA macro `SldprtToStep.bas` | the CAD person, inside SolidWorks | `manifest.csv` (no hashes) |
| B | SolidWorks Task Scheduler, "Export Files" | the CAD person, no macro | none (we validate the folder) |
| C | PowerShell batch `sldprt_to_step.ps1` | unattended or large runs | `manifest.json` + `.csv` with sha256 |

Operator sheet (Hebrew + English): `docs/production/runlog/cad_import/INSTRUCTIONS_SHERMAN.md`.

## A. VBA macro (Tools → Macro → Run)

You set it up once:
1. In SolidWorks, choose **Tools → Macro → New…** and save it as `SldprtToStep.swp`. The VBA editor opens.
2. In the VBA editor, choose **File → Import File…** and pick `SldprtToStep.bas`.
3. In the project tree, right-click the empty module SolidWorks created (`SldprtToStep1`) and choose
   **Remove** (answer *No* to export). Then choose **File → Save** and close the editor.

To run it, choose **Tools → Macro → Run** and pick `SldprtToStep.swp`. Then:
1. Pick the folder with the parts.
2. Pick an empty output folder.
3. Wait for the summary box.

The macro does the following:
- Sets STEP to AP214 and puts your own STEP setting back at the end.
- Exports in mm. The unit is changed in memory only; the part is never saved.
- Writes `<name>.step` into the output folder, mirroring the subfolders, and adds a row to `manifest.csv`.
- Runs again in the same output folder, appending rows. A STEP that is newer than its part is skipped.

## B. SolidWorks Task Scheduler: "Export Files"

This route needs no macro. Start **SOLIDWORKS Task Scheduler** from the Windows Start menu
(SOLIDWORKS Tools folder), then click **Export Files**:
- **Task files/folders:** click **Add Folder**, tick *Include subfolders*, and set the file type to **Part Files (*.sldprt)**.
- **Export to:** **STEP AP214 (*.step;*.stp)**. If the list offers only "STEP", click **Options…** and choose
  **AP214**. Leave the configuration option at the active configuration.
- **Output folder:** an empty folder. Do not choose "save in the same folder".
- **Schedule:** **Run Now** (or a time), then **Finish**.

Do **not** use "Convert Files". That task upgrades and saves the source parts.

Option labels differ a little between SolidWorks versions. Task Scheduler also writes no manifest. We
validate the returned folder with `validate --step-dir` instead.

## C. PowerShell batch (unattended, large runs)

This runs on stock Windows PowerShell 5.1 with SolidWorks installed and installs nothing. It reads the
API constants from the installed SolidWorks' own `api\redist\SolidWorks.Interop.swconst.dll`.

Close SolidWorks first, then run this one line:

```powershell
powershell -ExecutionPolicy Bypass -File sldprt_to_step.ps1 -Source D:\CAD\Parts -Output D:\CAD\STEP_out
```

Options:
- `-ListFile list.txt`: one file or folder per line; `#` starts a comment.
- `-DryRun`: plan only, no SolidWorks; writes `_logs\manifest.dryrun.*`.
- `-Force`: re-export files that are up to date.
- `-AllConfigurations`: writes `<name>__<config>.step`.
- `-TimeoutSec 300`: per file. On a hang SolidWorks is killed and restarted, and the file is retried once,
  then marked `TIMEOUT`.
- `-RestartEvery 25`: restart SolidWorks every N files to cap memory.
- `-LogDir`: default `<Output>\_logs`.
- `-AttachRunning`: use an already open SolidWorks. A hang still kills it.
- `-SwConstDll`: path to the constants DLL when SolidWorks is not under `C:\Program Files`.

Per file, the batch does the following:
1. Hashes the source (sha256) before and after; a difference is `SOURCE_CHANGED`.
2. Opens the part with `OpenDoc6` Silent|ReadOnly and records open warnings and rebuild "what's wrong" counts.
3. Exports with `Extension.SaveAs` (Silent|Copy) to `<name>.~tmp.step`, then renames it atomically.
4. Closes everything with discard (`CloseAllDocuments(true)`).

Other rules:
- `.SLDASM` and `.SLDDRW` files get the status `UNSUPPORTED_TYPE`.
- Paths containing `_SEALED` are skipped and only counted, never listed.
- Statuses: `OK`, `SKIPPED_UP_TO_DATE` (same source sha256 and the recorded STEP is still present and
  unchanged), `FAILED_OPEN`, `FAILED_EXPORT`, `TIMEOUT`, `UNSUPPORTED_TYPE`, `SOURCE_CHANGED`.
- Exit code: 0 for a clean run, 1 if anything failed, 2 for bad arguments, 3 when it refuses (SolidWorks
  or its constants are missing, or SolidWorks is already running).
- If a run is killed, `swStepAP` is restored at the next start from `_logs\steppref.restore.json`.

Scheduling (optional): run this once as the user who stays logged on. SolidWorks needs a desktop session.

```powershell
powershell -ExecutionPolicy Bypass -File register_task.ps1 -Source D:\CAD\Parts -Output D:\CAD\STEP_out -At 02:00
```

## Back at our side (Mac or station)

```bash
OMP_NUM_THREADS=1 nice -n 15 .venv/bin/python -m apps.production_line.recipes.cad_import validate \
    --manifest RETURNED/manifest.csv --out RETURNED/validation.json      # or --step-dir RETURNED
OMP_NUM_THREADS=1 nice -n 15 .venv/bin/python -m apps.production_line.recipes.cad_import import \
    --manifest RETURNED/manifest.csv --data-root <line data root> --catalog-db <catalog sqlite> \
    --upload-dir <uploads dir> --out RETURNED/import.json
```

Import creates draft catalog parts only, with the SKU registered and the CAD attached unanalyzed. Release
stays a human step on the setup screen.

## Not verified on this Mac

These parts have never run against SolidWorks; see `docs/production/runlog/cad_import/REPORT.md`:
- the COM calls;
- the enum names;
- the `[ref]` out-parameters;
- `SaveAs` with `$null` export data;
- the unit switch;
- watchdog kill and restart;
- the VBA macro.
