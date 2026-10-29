#!/usr/bin/env python3
"""Run one engine lane fresh on one scan and write evaluator-format predictions.

Lanes
  best_f1  The lane behind the historical Gold11 line-F1 0.4580152672: UNION of
           (A0) strict-blind scan-only typed union, and (CAD) blind CAD+PLY runtime
           detections that pass the July raw-scan support gate. Steps, each its own
           monitored process:
             1 decompose  backend.patch_graph_latent_decomposition (part_id=None, route apply)
             2 pool       scripts/build_f1_bounded_pool_from_freeze.build_one (voxel 1.8 mm)
             3 a0_union   scripts/run_f1_structured_union_e2e.py --disable-birth
                          (CP-SAT in .venv-f1-structured)
             4 cad        scripts/evaluate_bend_case.py --inference-mode blind_cad_ply
                          --skip-heatmap-consistency   (only with --cad)
             5 gate+fuse  in process (to_eval_format.convert_cad_case + fuse)
           Without --cad the lane degrades to A0 only (status A0_ONLY_NO_CAD).
  runtime  The active runtime exactly as the API launches it:
           backend/bend_analysis_worker.py (shared alignment, run_progressive_bend_inspection,
           tolerances 1.0 deg / 0.5 mm, PYTHONPATH=ROOT:ROOT/backend). Requires --cad.
           Detected lines are mapped to the raw-scan frame (to_eval_format runtime_report).

Blindness: the scan (and CAD) are copied into the work dir under opaque names
(scan_<sha12>.<ext>) and the runtime gets an opaque part id, so no filename,
part number or label reaches inference. --no-stage disables this.

Resources: before every heavy step the total RSS of python processes must be
< --rss-limit-gib (default 9) or the step waits; each step runs under
/usr/bin/time -l in its own process group, SIGKILLed after
min(--step-timeout, remaining --lane-timeout) seconds (both default 600). Time spent waiting for
the lock or at the RSS gate does not consume the lane budget. An flock on --lock serializes
broad-baseline heavy steps across processes. Each lane run first clears its own previous
prediction files in --out, and a failed step writes the failure to every variant file.

Outputs (in --out): predictions_<lane>.json (+ component lanes), run_record.json,
logs/. Bulky intermediates go to --work-dir (default <out>/work).

Example
  .venv/bin/python -B research/broad_baseline/run_lane.py --lane best_f1 \
      --scan ~/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.ply \
      --cad ~/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.stp \
      --out docs/production/runlog/broad_baseline/runs/<scan_id>/best_f1
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bb_common as bb  # noqa: E402
import to_eval_format as tef  # noqa: E402

DEFAULT_LOCK = Path("/private/tmp/claude-501/sherman_broad_baseline_engine.lock")


@contextlib.contextmanager
def _engine_lock(path: Path, *, poll_s: float = 5.0, max_wait_s: float = 7200.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    started = time.monotonic()
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() - started > max_wait_s:
                handle.close()
                raise RuntimeError(f"engine lock {path} busy for {max_wait_s}s")
            time.sleep(poll_s)
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "since": bb.utc_now()}))
        handle.flush()
        yield {"lock": str(path), "lock_wait_s": round(time.monotonic() - started, 1)}
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class Lane:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.scan_src = bb.expand(args.scan)
        self.cad_src = bb.expand(args.cad) if args.cad else None
        self.out = bb.expand(args.out)
        self.work = bb.expand(args.work_dir) if args.work_dir else self.out / "work"
        self.logs = self.out / "logs"
        self.started = time.monotonic()
        # Time spent waiting for the engine lock or for the python-RSS start gate (other jobs' memory)
        # is not compute and must not consume this lane's budget.
        self.excluded_wait_s = 0.0
        self.steps: list[dict[str, Any]] = []
        self.scan_sha = bb.sha256_file(self.scan_src)
        self.cad_sha = bb.sha256_file(self.cad_src) if self.cad_src else None
        self.token = "BB" + self.scan_sha[:12]

    # -- budget -------------------------------------------------------------
    def remaining(self) -> float:
        step = float(self.args.step_timeout)
        if float(self.args.lane_timeout) <= 0:
            return step
        left = float(self.args.lane_timeout) - (time.monotonic() - self.started - self.excluded_wait_s)
        return max(1.0, min(step, left))

    def heavy(self, name: str, cmd: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
        # Wait at the RSS gate BEFORE computing the step timeout, and exclude that wait from the lane
        # budget (run_monitored re-checks the gate, which then passes immediately).
        t0 = time.monotonic()
        pre_gate = bb.wait_for_rss_budget(float(self.args.rss_limit_gib))
        self.excluded_wait_s += time.monotonic() - t0
        record = bb.run_monitored(cmd, name=name, log_dir=self.logs, env=env, timeout_s=self.remaining(),
                                  rss_limit_gib=float(self.args.rss_limit_gib))
        record["pre_step_rss_gate"] = pre_gate
        record["lane_excluded_wait_s_so_far"] = round(self.excluded_wait_s, 1)
        self.steps.append(record)
        return record

    # -- staging ------------------------------------------------------------
    LANE_OUTPUTS = {
        "best_f1": ("predictions_best_f1.json", "predictions_a0.json", "predictions_cad_gated.json",
                    "predictions_cad_raw_ungated.json", "predictions_f1_zero_shot_decomposition.json"),
        "runtime": ("predictions_runtime.json", "predictions_runtime_gated.json",
                    "predictions_runtime_with_placeholders.json"),
    }

    def clear_previous_outputs(self) -> None:
        """A re-run into an existing --out must not leave an earlier run's predictions behind."""
        removed = []
        for name in (*self.LANE_OUTPUTS[self.args.lane], "run_record.json"):
            target = self.out / name
            if target.is_file():
                target.unlink()
                removed.append(name)
        if removed:
            self.steps.append({"name": "clear_previous_outputs", "status": "OK", "removed": removed})

    def stage(self) -> None:
        staged = self.work / "staged"
        staged.mkdir(parents=True, exist_ok=True)
        if self.args.no_stage:
            self.scan = self.scan_src
            self.cad = self.cad_src
        else:
            self.scan = staged / f"scan_{self.scan_sha[:12]}{self.scan_src.suffix.lower()}"
            if not self.scan.exists() or bb.sha256_file(self.scan) != self.scan_sha:
                shutil.copyfile(self.scan_src, self.scan)
            self.cad = None
            if self.cad_src is not None:
                self.cad = staged / f"cad_{self.cad_sha[:12]}{self.cad_src.suffix.lower()}"
                if not self.cad.exists() or bb.sha256_file(self.cad) != self.cad_sha:
                    shutil.copyfile(self.cad_src, self.cad)
        if bb.sha256_file(self.scan) != self.scan_sha:
            raise RuntimeError("staged scan sha256 mismatch")
        self.steps.append({"name": "stage", "status": "OK", "scan": str(self.scan), "cad": None if self.cad is None
                           else str(self.cad), "opaque": not self.args.no_stage})

    # -- best_f1 components ---------------------------------------------------
    def run_a0(self) -> tuple[dict[str, Any], dict[str, Any]]:
        py = str(bb.APP_PYTHON)
        steps_py = str(bb.HERE / "bb_steps.py")
        pool_scan, pool_sha = self.scan, self.scan_sha
        if self.scan.suffix.lower() != ".ply":
            proxy = self.work / "staged" / f"scan_{self.scan_sha[:12]}__ply_proxy.ply"
            rec = self.heavy("a0_ply_proxy", [py, "-B", steps_py, "ply-proxy", "--scan", str(self.scan),
                                              "--out", str(proxy)])
            if rec["status"] != "OK":
                return self._a0_failed("PLY_PROXY_FAILED", rec)
            pool_scan, pool_sha = proxy, bb.sha256_file(proxy)
        decomposition = self.work / "a0" / self.token / "result" / "decomposition_result.json"
        # Non-PLY scans: decompose the lossless PLY proxy too. build_one() requires the decomposition's
        # scan to hash- or geometry-match the pool source, and open3d's OBJ/STL vertex view does not
        # geometry-match the trimesh proxy (broad run 2026-10-17, sc_8379cebef9e3 OBJ).
        rec = self.heavy("a0_decompose", [py, "-B", steps_py, "decompose", "--scan", str(pool_scan),
                                          "--out-json", str(decomposition),
                                          "--cache-dir", str(self.work / "a0_cache")])
        if rec["status"] != "OK" or not decomposition.is_file():
            return self._a0_failed("DECOMPOSITION_FAILED", rec)
        decomp_eval = tef.convert_decomposition(bb.read_json(decomposition), source=decomposition, scan=self.scan_src,
                                                proxy_sha256=pool_sha if pool_scan != self.scan else None)
        bb.write_json(self.out / "predictions_f1_zero_shot_decomposition.json", decomp_eval)
        pool_dir = self.work / "pool"
        rec = self.heavy("a0_pool", [py, "-B", steps_py, "pool", "--decomposition-root", str(self.work / "a0"),
                                     "--token", self.token, "--source-scan", str(pool_scan),
                                     "--source-sha", pool_sha, "--out-dir", str(pool_dir), "--voxel-mm", "1.8"])
        pool = pool_dir / f"{self.token}.json"
        if rec["status"] != "OK" or not pool.is_file():
            return self._a0_failed("CANDIDATE_POOL_FAILED", rec, decomp_eval)
        union_dir = self.work / "a0_union"
        rec = self.heavy("a0_typed_union", [py, "-B", str(bb.ROOT / "scripts" / "run_f1_structured_union_e2e.py"),
                                            "--candidate-pool", str(pool), "--output-dir", str(union_dir),
                                            "--disable-birth", "--solver-python", str(self.args.solver_python)])
        prediction = union_dir / "prediction_owned_regions.json"
        if rec["status"] != "OK" or not prediction.is_file():
            return self._a0_failed("TYPED_UNION_FAILED", rec, decomp_eval)
        a0_eval = tef.convert_a0(bb.read_json(prediction), source=prediction, scan=self.scan_src)
        bb.write_json(self.out / "predictions_a0.json", a0_eval)
        return a0_eval, decomp_eval

    def _a0_failed(self, reason: str, rec: dict[str, Any], decomp: dict[str, Any] | None = None):
        detail = {"step": rec.get("name"), "returncode": rec.get("returncode"), "timed_out": rec.get("timed_out"),
                  "stderr_log": rec.get("stderr_log")}
        a0 = tef.empty_lane("a0_scan_only_typed_union", scan=self.scan_src, reason=reason, detail=detail)
        bb.write_json(self.out / "predictions_a0.json", a0)
        if decomp is None:
            decomp = tef.empty_lane("f1_zero_shot_decomposition", scan=self.scan_src, reason=reason, detail=detail)
            bb.write_json(self.out / "predictions_f1_zero_shot_decomposition.json", decomp)
        return a0, decomp

    def run_cad(self) -> dict[str, Any]:
        if self.cad is None:
            cad = tef.empty_lane("cad_gated", scan=self.scan_src, reason="NO_CAD_SUPPLIED")
            cad["status"] = "LANE_SKIPPED_NO_CAD"
            bb.write_json(self.out / "predictions_cad_gated.json", cad)
            return cad
        case_json = self.work / "cad" / "cad_case.json"
        case_json.parent.mkdir(parents=True, exist_ok=True)
        identity = f"{self.cad.resolve()}\0{self.scan.resolve()}"
        blind_id = "blind-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        rec = self.heavy("cad_blind_runtime", [
            str(bb.APP_PYTHON), "-B", str(bb.ROOT / "scripts" / "evaluate_bend_case.py"),
            "--part-key", self.token, "--inference-part-id", blind_id,
            "--cad-path", str(self.cad), "--scan-path", str(self.scan), "--output-json", str(case_json),
            "--state", "unknown", "--expected-total", "None", "--expected-completed", "None",
            "--spec-bend-angles-json", "[]", "--spec-schedule-type", "",
            "--inference-mode", "blind_cad_ply", "--skip-heatmap-consistency",
        ])
        if rec["status"] != "OK" or not case_json.is_file():
            cad = tef.empty_lane("cad_gated", scan=self.scan_src, reason="CAD_RUNTIME_FAILED",
                                 detail={"returncode": rec.get("returncode"), "timed_out": rec.get("timed_out"),
                                         "stderr_tail": _tail(rec.get("stderr_log"))})
            bb.write_json(self.out / "predictions_cad_gated.json", cad)
            bb.write_json(self.out / "predictions_cad_raw_ungated.json", {**cad, "lane": "cad_raw_ungated"})
            return cad
        payload = bb.read_json(case_json)
        t0 = time.monotonic()
        cad = tef.convert_cad_case(payload, source=case_json, scan=self.scan_src, gate=True)
        raw = tef.convert_cad_case(payload, source=case_json, scan=self.scan_src, gate=False)
        self.steps.append({"name": "cad_gate", "status": "OK", "wall_s": round(time.monotonic() - t0, 3),
                           "accepted": cad["line_count"], "rejected": len(cad["dropped"])})
        bb.write_json(self.out / "predictions_cad_gated.json", cad)
        bb.write_json(self.out / "predictions_cad_raw_ungated.json", raw)
        return cad

    def best_f1(self) -> dict[str, Any]:
        # CAD half first: it is seconds, while the exact CP-SAT typed union can take minutes and
        # must not starve it of the shared lane budget.
        cad = self.run_cad()
        a0, _decomp = self.run_a0()
        fused = tef.fuse(a0, cad, mode="union")
        if a0["status"] == "LANE_FAILED" and cad["status"] in {"LANE_FAILED", "LANE_SKIPPED_NO_CAD"}:
            fused["status"] = "LANE_FAILED"
            fused["failure"] = {"a0": a0.get("failure"), "cad": cad.get("failure")}
        elif a0["status"] == "LANE_FAILED":
            fused["status"] = "DEGRADED_CAD_ONLY_A0_FAILED"
        elif cad["status"] == "LANE_SKIPPED_NO_CAD":
            fused["status"] = "A0_ONLY_NO_CAD"
        elif cad["status"] == "LANE_FAILED":
            fused["status"] = "DEGRADED_A0_ONLY_CAD_FAILED"
        fused["lane"] = "best_f1"
        fused["component_status"] = {"a0": a0["status"], "a0_failure": a0.get("failure"),
                                     "cad_gated": cad["status"], "cad_failure": cad.get("failure")}
        bb.write_json(self.out / "predictions_best_f1.json", fused)
        return fused

    # -- runtime ----------------------------------------------------------------
    def runtime(self) -> dict[str, Any]:
        if self.cad is None:
            lane = tef.empty_lane("runtime", scan=self.scan_src, reason="RUNTIME_REQUIRES_CAD")
            bb.write_json(self.out / "predictions_runtime.json", lane)
            return lane
        out_dir = self.work / "runtime"
        out_dir.mkdir(parents=True, exist_ok=True)
        env = {"PYTHONPATH": os.pathsep.join([str(bb.ROOT), str(bb.ROOT / "backend")]),
               "BEND_PHASE_LOG_PATH": str(out_dir / "bend_worker.phase.log")}
        rec = self.heavy("runtime_worker", [
            str(bb.APP_PYTHON), "-B", str(bb.ROOT / "backend" / "bend_analysis_worker.py"),
            "--job-id", self.token, "--cad-path", str(self.cad), "--scan-path", str(self.scan),
            "--part-id", self.token, "--part-name", self.token,
            "--tolerance-angle", "1.0", "--tolerance-radius", "0.5", "--output-dir", str(out_dir),
        ], env=env)
        report = out_dir / "bend_inspection_report.json"
        error = _worker_error(rec.get("stdout_log"))
        if rec["status"] != "OK" or not report.is_file():
            lane = tef.empty_lane("runtime", scan=self.scan_src, reason="RUNTIME_WORKER_FAILED",
                                  detail={"returncode": rec.get("returncode"), "timed_out": rec.get("timed_out"),
                                          "worker_error": error, "stderr_tail": _tail(rec.get("stderr_log"))})
            for name in self.LANE_OUTPUTS["runtime"]:  # every variant scores the failure explicitly
                bb.write_json(self.out / name, {**lane, "lane": name.removeprefix("predictions_").removesuffix(".json")})
            return lane
        payload = bb.read_json(report)
        t0 = time.monotonic()
        lane = tef.convert_runtime_report(payload, source=report, scan=self.scan_src)
        with_ph = tef.convert_runtime_report(payload, source=report, scan=self.scan_src, include_placeholders=True)
        gated = tef.convert_runtime_report(payload, source=report, scan=self.scan_src, gate=True)
        self.steps.append({"name": "runtime_convert", "status": "OK", "wall_s": round(time.monotonic() - t0, 3),
                           "lines": lane["line_count"], "placeholders_dropped": len(lane["dropped"])})
        bb.write_json(self.out / "predictions_runtime.json", lane)
        bb.write_json(self.out / "predictions_runtime_with_placeholders.json", with_ph)
        bb.write_json(self.out / "predictions_runtime_gated.json", gated)
        return lane

    # -- record -----------------------------------------------------------------
    def record(self, result: dict[str, Any], lock_info: dict[str, Any]) -> dict[str, Any]:
        heavy = [s for s in self.steps if "wall_s" in s and "command" in s]
        peak = max([max(s.get("time_max_rss_bytes") or 0, s.get("sampled_peak_group_rss_bytes") or 0)
                    for s in heavy] or [0])
        record = {
            "schema": "sherman.broad_baseline.run_record.v1",
            "lane": self.args.lane,
            "status": result.get("status"),
            "line_count": result.get("line_count"),
            "scan": {"source": str(self.scan_src), "sha256": self.scan_sha, "staged": str(self.scan)},
            "cad": None if self.cad_src is None else {"source": str(self.cad_src), "sha256": self.cad_sha,
                                                       "staged": str(self.cad)},
            "opaque_token": self.token,
            "work_dir": str(self.work),
            "lane_wall_s": round(time.monotonic() - self.started, 3),
            "lane_excluded_wait_s": round(self.excluded_wait_s, 1),
            "heavy_step_wall_s": round(sum(s["wall_s"] for s in heavy), 3),
            "peak_rss_bytes": peak,
            "limits": {"step_timeout_s": self.args.step_timeout, "lane_timeout_s": self.args.lane_timeout,
                       "rss_limit_gib": self.args.rss_limit_gib},
            "steps": self.steps,
            "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                            "app_python": str(bb.APP_PYTHON), "solver_python": str(self.args.solver_python)},
            **lock_info,
            "finished_at": bb.utc_now(),
        }
        bb.write_json(self.out / "run_record.json", record)
        return record


def _tail(path: str | None, lines: int = 25) -> str | None:
    if not path or not Path(path).is_file():
        return None
    text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def _worker_error(path: str | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("ERROR\t"):
            try:
                return json.loads(line.split("\t", 1)[1]).get("error")
            except json.JSONDecodeError:
                return line
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lane", required=True, choices=("best_f1", "runtime"))
    parser.add_argument("--scan", required=True)
    parser.add_argument("--cad", help="CAD (STEP/IGES or mesh); required for runtime and the CAD half of best_f1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--work-dir", help="bulky intermediates (pool ~10 MB, graphs ~15 MB); default <out>/work")
    parser.add_argument("--step-timeout", type=float, default=600.0)
    parser.add_argument("--lane-timeout", type=float, default=600.0, help="total budget per lane run; 0 = none")
    parser.add_argument("--rss-limit-gib", type=float, default=9.0)
    parser.add_argument("--solver-python", default=str(bb.SOLVER_PYTHON))
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--no-stage", action="store_true", help="run on the original paths (filenames visible)")
    args = parser.parse_args(argv)
    lane = Lane(args)
    lane.out.mkdir(parents=True, exist_ok=True)
    with _engine_lock(Path(args.lock)) as lock_info:
        lane.excluded_wait_s += float(lock_info.get("lock_wait_s") or 0.0)
        lane.clear_previous_outputs()
        lane.stage()
        result = lane.best_f1() if args.lane == "best_f1" else lane.runtime()
        record = lane.record(result, lock_info)
    print(json.dumps({"lane": args.lane, "status": record["status"], "lines": record["line_count"],
                      "lane_wall_s": record["lane_wall_s"], "peak_rss_gib": round(record["peak_rss_bytes"] / 2**30, 3),
                      "out": str(lane.out)}))
    return 0 if record["status"] not in {"LANE_FAILED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
