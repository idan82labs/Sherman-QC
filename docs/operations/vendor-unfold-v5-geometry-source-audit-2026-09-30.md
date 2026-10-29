# V5 geometry source audit — 2026-09-30

Disposition: **GREEN_FOR_SOURCE_FREEZE_AND_INDEPENDENT_PRE** for the source
revisions below. This is a source/pure-synthetic review of the new finite-strip
mechanism and its connection to V5. It grants no physical execution, preview
promotion, corpus accuracy, continuous trim proof or manufacturing authority.
The later frozen source bundle and independent PRE must match these revisions
or explicitly audit subsequent changes.

The reviewer did not author the local-strip helper, its diagnostic add-on, V5
integration or extractor V2. The same reviewer authored the independent oracle
and bridge; those files are connected checks here, not independently audited
by their own author. Their independent source/receipt audit remains a separate
gate.

## Reviewed source

| Source | SHA-256 |
| --- | --- |
| `backend/vendor_unfold_local_strip_guard_v1.py` | `7268fede9a8817b86a18f749d6fb682088a709a5aae4fc0c31b6c00066c14fad` |
| `backend/vendor_unfold_preview_v5.py` | `05e5004296bc1555e1cecba91f8435e5550e1d5d9855f96efdebf54d533081c7` |
| `backend/vendor_unfold_local_strip_contract_v1.py` | `4bc09c868de658bf516a97a75a01088dbde90bfb7a4caa409b8b6885b75ae45e` |
| `backend/vendor_unfold_contract_v4.py` | `48bd939808588172c635c6d59d0eaa4156e54c326dc243d46325d48f474ef18c` |
| `backend/vendor_unfold_native_v5.py` | `ce652f0fa13f1566282b070a1651b2f219d1b7aab642e065e3b9067dad30e89f` |
| `scripts/extract_step_topology_bend_axes_v2.py` | `8217ffb09c46e04f3840b8dc687ad72014be6101ad2d9e8f6f85d8b95be13f65` |

No physical inputs, retained case artifacts, manifests or control records were
opened in this review. Native geometry experiments were not run during this
audit. The earlier independently constructed native bridge checks are not
relabelled as tests of these later source bytes.

## Mathematical model and finding

For a selected cylindrical skin with radius R and actual native seam sweep
delta, its intrinsic developed width is W = R * abs(delta). In the parent
flange chart, signed tangent distance points toward the bend strip. In the
child chart, that tangent is transported by the actual native rotation and
its polarity is inverted. This is consistent with the proper rotation used
to develop the child flange: the cylinder occupies [0,W] when viewed from
either seam with the corresponding polarity.

The new helper computes axial and signed intrusion coordinates as affine
functions of every supplied source vertex. It clips every triangle against
the expanded axial interval and the finite intrusion strip using exact
Fractions of the supplied binary floats. A convex triangle clipped by four
halfplanes has at most seven vertices. A nonzero exact shoelace area after
clipping witnesses positive-area material intrusion and refuses. Vertex-only
tests cannot establish this fact; triangles crossing the strip with all
original vertices outside are included.

The lower strip side is the incumbent tolerance minus charged seam/frame
uncertainty, including the conservative tangent norm correction. The upper
side extends W by tolerance and uncertainty. Serialized intrusion bounds are
a subset of the exact checked strip: lower rounded upward, upper downward.
They therefore do not claim unchecked one-ULP slivers. The frame residual
multiplied by source scale is capped before certification; arithmetic at
insufficient coordinate resolution and excessive rational complexity refuses.

A full native seam must also be witnessed by mesh boundary intervals. An
incident triangle must contain definite material on the expected local side,
and its seam edge must have exactly one triangle owner. The intervals must
cover the entire native finite axial span within the same numerical tolerance.
Remote material beyond W may be retained. This is safe only together with the
unchanged whole-face development overlap guard and the independent native
source/edge oracle. The helper alone does not prove continuous trimmed CAD
material, native seam identity, connected sheet topology or manufacturing
blank dimensions. Degenerate source geometry must be rejected by the native
mesh, exported-mesh and oracle gates; this helper is not standalone mesh
admission.

The clipping construction follows the halfplane re-entry model of
[Sutherland and Hodgman, Reentrant polygon clipping (1974)](https://doi.org/10.1145/360767.360802).
The publisher full text was unavailable to this review; the proof above was
derived independently from convexity and affine coordinates, and the code was
checked against an independently written exact rectangle-intersection oracle.
Native U/V range and derivative interpretation was checked against the
[OCCT BRepAdaptor_Surface reference](https://occt3d.com/dev/doc/refman/html/class_b_rep_adaptor___surface.html).

## V5 binding and connected checks

V5 wraps the unchanged native `_child_transform` and records its returned
delta and cylinder identity immediately before parent/child certificates are
computed. Strip width is computed from that cylinder radius and the absolute
returned delta. It is not a caller-selected clearance or principal-angle
replacement. Parent and child share one fixed `LocalStripBudget` for the
complete development. The reused component retains proper-rotation,
native-seam closure, full source-material role checks and whole unchanged
developed-mesh overlap.

The diagnostic add-on checks radius/sweep/width binding, parent-child equality,
exported sweep consistency, declared uncertainty and conservative serialized
strip bounds. It reconstructs the complete parent/child work chronology from
positive cumulative triangle counts, checks tree ownership and consumes all
local work under the fixed caps. These declaration checks do not cryptographically
prove runtime identity; the frozen source/runtime pins and worker supervision
remain mandatory.

The common mesh/source/marker validator uses a detached declaration view. The
original V5 payload retains its actual local-strip model and coplanar support;
the new add-on and source-slice/winding validator then inspect that original
payload. No incumbent finite-width certificate is forged or admitted as a
local-strip proof.

The V5 native reader now calls extractor V2 on the private source snapshot and
passes every native face through the evidence observer before development.
After development, the bridge must succeed before the result is returned.
Failure returns an empty unavailable envelope. The oracle verifies original
face/triangle lineage, native oriented-normal parity, all selected shared-edge
polygon nodes, per-face intrinsic metric and polygon area, all bend ownership
and finite lines. The model remains discrete native tessellation; continuous
area is explicitly NOT_ESTABLISHED unless a separately justified model exists.

## Extractor V2 representation change

The lower 15-degree eligibility boundary is now evaluated in the same
six-decimal degree representation already published to consumers. The raw
200-degree upper limit and raw 8-mm axial span limit are unchanged. This admits
at most the half-microdegree publication rounding neighborhood at the lower
boundary; it is a declared representation policy, not a manufacturing
tolerance. The raw sweep, published sweep and owning native face ID are all
retained in `native_sweep_rows`, and the existing coaxial/radius grouping and
native signed-V endpoint construction remain unchanged.

V5/upload continues to impose its tighter 175-degree maximum and paired sweep
consistency. Full cylinders therefore remain excluded. Split or unexplained
material retains an explicit refusal. The legacy `canonical_line_rule` string
still says `NEUTRAL_RADIUS_HALF_ANGULAR_TURN`; the actual computation is an
average geometric skin midpoint and carries no neutral-axis/K-factor proof.
That inherited naming is a nonblocking semantic limitation for this visual
preview and must not be promoted into manufacturing documentation.

## Executed evidence and corrected finding

- Independent exact Fraction rectangle oracle: 36 Cartesian cases covering
  inside/outside/crossing rectangles, expanded axial boundaries and one-ULP
  intrusion slivers. Helper decisions matched the independent positive-area
  rectangle-intersection calculation in all 36 cases.
- Independent C-return probe with both triangle windings: certification was
  unchanged; every source triangle remained charged. Work was 8 triangles,
  16 projected vertices, 71 clip-edge tests and 24 seam-edge tests.
- Independent diagnostic mutations of radius, sweep, width and intrusion
  bounds all refused; a changed exported sweep also refused.
- An intermediate integration lacked three required local-strip limitations,
  producing 19 failed / 18 passed focused checks. Root corrected the envelope;
  the failure is retained here and is not described as a geometry result.
- On the source revisions listed above, the focused source-only command below
  returned exit 0: **64 passed in 0.68s**.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -c /dev/null --noconftest -p no:cacheprovider -q tests/test_vendor_unfold_local_strip_guard_v1.py tests/test_vendor_unfold_preview_v5.py tests/test_vendor_unfold_oracle_bridge_v1.py tests/test_extract_step_topology_bend_axes_v2.py
```

No blocking source finding remains in this review's scope. Independent frozen
PRE, the single bounded physical successor, independent POST and visual QA
remain required before reporting physical preview coverage or promotion.
