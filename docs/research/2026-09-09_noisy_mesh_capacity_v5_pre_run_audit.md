# Noisy-mesh capacity V5: independent pre-run audit

Date: 2026-09-09. Root independently read all407 test-source lines and225 protocol
lines after prospective fixture approval and before any execution. Author
`/root/cad_support_lineage` did not run the geometry or tests. Root audited the
constructor, label-independent dyadic noise, support/witness/cone and resource
oracles, and the separate source/full-capacity harness routes.

## Exact six-file closure

| File | SHA-256 |
| --- | --- |
| `backend/cad_guided_noisy_mesh_planes_v5.py` | `9a045e2f7c7fe5fc21c38cf169236ea7d3fa181f5d8298c6b9882935b3793b75` |
| `docs/research/2026-09-09_noisy_mesh_planes_v5_protocol.md` | `039b6a3f46e170947039ee32a8ce871f768aea781b369b244fc0e96a9965b71b` |
| `docs/research/2026-09-09_noisy_plane_registration_design_notes.md` | `03e014332e4a9072f631dd449c6c6f55710fba175dbe19af6e7349eed7562e52` |
| `tests/test_cad_guided_noisy_mesh_capacity_v5.py` | `48f4f99537c2456cb40a53ab6165f5c608c75e707274153f42e5a7d079d68795` |
| `docs/research/2026-09-09_noisy_mesh_capacity_v5_protocol.md` | `cd446f96a086ac820630df1aa796ea40fad827c0cc88883af843e7756c09b1c5` |
| `scripts/check_cad_guided_measurement.py` | `da3828c14c38f6eeabe1abadd00b7f455ebf421f0526abe76e25c597860237b7` |

The unchanged extractor already passed75 independently authored controls with
source custody. That is prerequisite evidence, not proof of this larger case.
The prior search-gate harness b11c0185 is byte-preserved under
`source_snapshots/plane_search_v5_b11c0185/` before these additive routes.

## Oracle audit

The exact full construction is900,000 used raw rows /1,800,000 original triangles,
with2,700,000 shared undirected edges and5,400,000 extractor edge records. The
annular prism has shared raw seams, ten finite analytic faces and an actual hole;
the small160-row constructor control independently checks manifold incidence,
opposite winding, connectivity and Euler characteristic zero. The actual small
extractor prerequisite has18,304 rows/36,608 triangles; the full case is not
executed under its smaller source-gate budget.

The three integer-noise numerator maxima31/33/35 yield squared displacement at
most3275/8192^2, strictly below.007^2 and the unchanged stipulated.05mm error
radius. Integer truth plus dyadic perturbation is exactly representable here.
Every coordinate is perturbed; every face must have real normal noise. Labels
and analytic normals are evaluator-only and never passed to extraction.

Both full base partitions, all retained immutable original row unions/digests,
all qualified support witnesses and every qualified pure-face normal cone are
checked. Ten separate >=50-row qualified core representatives are required,
not a whole-plane count. Rejected/mixed cores, rejected closures and unscheduled
expansions are retained. No oracle loosening or epsilon increase is allowed.

## Gate decisions before execution

**GREEN now for one source/prerequisite run only:**

```sh
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite noisy_mesh_capacity_v5_source
```

60s wall /45s worker CPU /1GiB sampled group RSS /1MiB combined output. Expected
success is two passes and one declared large-case skip. The parent uses a fresh
allowlisted environment without the large opt-in flag. Source/receipt drift,
test failure, resource stop or monitoring failure blocks the full gate.

**CONDITIONAL only**, after root inspects that exact terminal successful receipt
and rechecks this closure, one serial full-capacity run may use:

```sh
.venv/bin/python -I -S -B scripts/check_cad_guided_measurement.py --suite noisy_mesh_capacity_v5
```

The harness sets the explicit flag only on this route. Limits stay120s wall /
90s worker CPU /4GiB sampled group RSS /1MiB output, including construction,
extraction and the full oracle. Exactly one large call runs; the two small
controls remain part of this gate. No other geometry job may overlap.

Retain every terminal outcome. Do not rerun a failure, resize the mesh, weaken
the ten-face requirement, raise the cap or relabel a skipped case as tested.
No physical input, native CAD, Gold, calibrated noise, registration, F1 or
production qualification is established by either permitted synthetic gate.

## Conditional full gate released after source receipt

Recorded before full-size dispatch: the source gate completed once in
`cad_engine_v1_synthetic_checks/noisy_mesh_capacity_v5_source-_09yfli0` with
two passes and the one declared large-case skip. Parent.603706583s,
worker CPU.44826s, sampled group peak98,107,392B and worker peak100,220,928B.
Output SHA `1a15f37335a6e0e7dfc387d00073bee23a4ae342e2b0e5c08c086434d8ea5e59`.
Root inspected the terminal receipt: no stop, no source drift; all six named
files were rehashed and match the table. No geometry process is still running.
The independently reviewed full gate is now **GREEN for its one fixed invocation**
under unchanged120/90-worker-CPU/4GiB/1MiB limits. This release predicts no outcome.
