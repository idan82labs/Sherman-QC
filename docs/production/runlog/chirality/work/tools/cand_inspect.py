"""E3 command for the E4 runner (--engine-cmd): runs the candidate tree work/repo_c's inspect CLI."""
import runpy, sys
W = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/chirality/work/repo_c"
sys.path[:0] = [W, W + "/backend"]
runpy.run_module("domains.production.engine.inspect", run_name="__main__", alter_sys=True)
