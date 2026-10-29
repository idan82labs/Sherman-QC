"""CP-SAT selection for Z5 (runs in .venv-f1-structured, which has ortools; stdlib + ortools only).

stdin/--in: JSONL, one problem per line:
  {"id": str, "w": [int weight per candidate, scaled x1000], "conflicts": [[i, j], ...],
   "atmost1": [[i, j, k], ...], "edges": [[i, a, b] (candidate i joins flange nodes a,b)],
   "n_nodes": int, "tree_slack": int or null, "nopair_penalty": int (subtracted for candidates not in edges)}
stdout/--out: JSONL {"id": str, "keep": [i, ...], "status": str}
Objective: maximise sum (w_i - penalty_i) x_i subject to pairwise conflicts, at-most-one groups and, when
tree_slack is set, a near-tree bound: (#selected flange edges) <= (#touched flange nodes) - 1 + tree_slack.
"""
import json
import sys

from ortools.sat.python import cp_model


def solve(p):
    n = len(p["w"])
    if n == 0:
        return {"id": p["id"], "keep": [], "status": "EMPTY"}
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(n)]
    for i, j in p.get("conflicts", []):
        m.AddBoolOr([x[i].Not(), x[j].Not()])
    for g in p.get("atmost1", []):
        if len(g) > 1:
            m.AddAtMostOne([x[i] for i in g])
    in_edge = set()
    edges = p.get("edges", [])
    if edges and p.get("tree_slack") is not None:
        z = [m.NewBoolVar(f"z{a}") for a in range(p["n_nodes"])]
        inc = {a: [] for a in range(p["n_nodes"])}
        for i, a, b in edges:
            in_edge.add(i)
            inc[a].append(i)
            inc[b].append(i)
            m.AddImplication(x[i], z[a])
            m.AddImplication(x[i], z[b])
        for a, lst in inc.items():  # a node is touched only if some selected edge uses it
            if lst:
                m.AddBoolOr([x[i] for i in lst] + [z[a].Not()])
            else:
                m.Add(z[a] == 0)
        ex = sorted(in_edge)
        anysel = m.NewBoolVar("any")
        m.AddMaxEquality(anysel, [x[i] for i in ex])
        m.Add(sum(x[i] for i in ex) <= sum(z) - anysel + int(p["tree_slack"]))
    else:
        in_edge = {i for i, _, _ in edges}
    pen = int(p.get("nopair_penalty", 0))
    m.Maximize(sum((int(p["w"][i]) - (0 if i in in_edge else pen)) * x[i] for i in range(n)))
    s = cp_model.CpSolver()
    s.parameters.num_search_workers = 1
    s.parameters.max_time_in_seconds = 10.0
    s.parameters.random_seed = 0
    st = s.Solve(m)
    keep = [i for i in range(n) if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) and s.Value(x[i])]
    return {"id": p["id"], "keep": keep, "status": s.StatusName(st)}


def main():
    args = sys.argv[1:]
    fin = open(args[args.index("--in") + 1]) if "--in" in args else sys.stdin
    fout = open(args[args.index("--out") + 1], "w") if "--out" in args else sys.stdout
    for line in fin:
        line = line.strip()
        if line:
            fout.write(json.dumps(solve(json.loads(line))) + "\n")
    fout.flush()


if __name__ == "__main__":
    main()
