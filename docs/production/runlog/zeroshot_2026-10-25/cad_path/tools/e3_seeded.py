"""DEV harness (no engine edit): run the E3 CLI with zs->CAD registration seeds injected.
usage: e3_seeded.py --seeds FILE.json --mode base|add|only -- <E3 CLI args>
  base : unpatched E3 (same process shape, for runtime parity)
  add  : seeds are appended to the coarse datum hypotheses of the proper pass (proper seeds) and of the mirrored
         pass (seeds computed from the mirrored zs lines); the full search still runs (contract-safe arm)
  only : DIAGNOSTIC: the coarse raster peaks are replaced by the seeds when there are any (search-time arm);
         hole-pair hypotheses, free-datum search, mirror check and everything after are unchanged
The seeds are rigid 4x4 raw->cad_model_mm transforms only; E3 receives no file name, id or metadata."""
import json, sys, time
from pathlib import Path
REPO = Path("/Users/idantal/ShermanQC/FullProject")
sys.path[:0] = [str(REPO), str(REPO / "backend")]
import numpy as np


def main():
    a = sys.argv[1:]
    k = a.index("--")
    opts, cli = a[:k], a[k + 1:]
    seeds = json.loads(Path(opts[opts.index("--seeds") + 1]).read_text())
    mode = opts[opts.index("--mode") + 1]
    from domains.production.engine.inspect import engine as E
    from domains.production.engine.inspect.registration import PoseHypothesis
    if mode != "base":
        state = {"mirrored": False}
        orig_reg, orig_coarse = E._register, E.coarse_hypotheses

        def reg(*args, mirrored, **kw):
            state["mirrored"] = mirrored
            return orig_reg(*args, mirrored=mirrored, **kw)

        def coarse(vox, part, model, cfg, *, mirrored, budget_left, segments=None, plane_seeds=False):
            if plane_seeds:
                return orig_coarse(vox, part, model, cfg, mirrored=mirrored, budget_left=budget_left,
                                   segments=segments, plane_seeds=True)
            Ts = seeds["mirrored" if state["mirrored"] else "proper"]
            sh = [PoseHypothesis(T=np.asarray(T, float), score=10.0 - 0.01 * i, segment=-1, family=0, height=0.0,
                                 mirrored=False, theta_deg=0.0, info={"seed": i}) for i, T in enumerate(Ts)]
            if mode == "only" and sh:
                if segments is None:
                    from domains.production.engine.inspect.registration import extract_planes
                    segments = extract_planes(vox, cfg, min_area=max(30.0, cfg.datum_min_area_fraction *
                                                                     float(part.panels[part.datum].area)), thickness=part.t)
                return sh, segments
            hyps, segs = orig_coarse(vox, part, model, cfg, mirrored=mirrored, budget_left=budget_left, segments=segments)
            return sh + hyps, segs

        E._register = reg
        E.coarse_hypotheses = coarse
    from domains.production.engine.inspect.cli import main as cli_main
    raise SystemExit(cli_main(cli))


if __name__ == "__main__":
    main()
