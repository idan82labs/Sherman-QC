# Random email STEP transfer — exact CAD bend match

2026-10-01. User authorized relevant email CAD access. Reproducible selection
from five connector-returned STEP attachments chose `MEC003744_02.step`, received
August 4, associated email part alias `50757000_02`. This is an older STEP
delivery, not the newest September 30 SLDPRT. Input SHA256
`0199d1289e9ff67daabb7b25f9d69de30805b2e81ad43006bb7b916f573eae25`, 657,890 bytes.
Not a duplicate of approved20 by hash/name; no claim about all historical
project exposure. No scans, drawings, inspection truth or unrelated attachments
were downloaded. Selection metadata exposed bend-count hints for two unselected
candidates, so the selection is not described as strictly blinded.

## Outcome

Unchanged frozen extractor V10 produced **12 true positives, zero false
positives, zero false negatives** against an independently frozen complete
technical reference: **precision, recall and exact-native-face F1 each 100%**.
Exact count 12/12. This is one CAD transfer case, not a generalization rate.
Matching uses the unchanged one-to-one complete native-face-union scorer.
No threshold, inference logic, case-ID behavior or reference changed from
feedback. The approved20 result remains 120/127 known capture (94.49%), not
silently merged with this case. No physical scan-vs-Gold F1 was run.

Neutral evidence has one valid solid, 278 faces (135 planar, 143 cylindrical)
and 750 edges, with no face/edge/pcurve/topology errors or coefficient refusals.
Reference review identified 12 finite quarter-turn bend pairs with nominal
inner/outer radii 1/4 and gauge 3. Each has opposite material orientation,
native tangent flange generator rails and two common quarter-annular end caps.
The other 119 cylindrical faces are 3-unit thickness-direction hole/rounded
cut walls, bounded by axis-normal planar faces. All 278 faces are classified;
unknown count zero under this stated geometric ontology. AI technical reviews
are not human manufacturing approval. Native units remain provisional.

## Execution evidence

Attempt: `artifacts/cad-email-transfer-v2-2026-10-01-attempt1/`.

| Gate/job | Pinned evidence / result |
| --- | --- |
| Full candidate freeze | `0d9e1d768b8da057dae79cc549fe5c53f4b06ad457e0c92eec4b38fce996b30f` |
| Neutral PRE | `f0ae7ba9a68c1de205de1dbbcb6fd655d84724bc850d2778a98549a0d020e2e5` |
| Neutral POST | `d280ab6435925a471c3faaf6351b72de2b83ff23e00f5be780aea08ad87c4f26` |
| Neutral job | outer0, cleanup known, 3.6096 s, peak sampled535712 KiB |
| Complete reference | `9ffac1ab2db7d38f471f5ac081a69c7ef6e932f1c255bd7e1acf6fa90f6b8873` |
| Extraction PRE | `2fa624fb6ca38e1ec9ac9fcfb4374ffb21975bccc679ef7e30e544850f75ff75` |
| Extraction POST | `4cc2ceae2cea492c74030d6e3765441477ead00fc35120ed62321527fd2ac7d9` |
| Extraction job | outer0, cleanup known, 1.1995 s, peak sampled512736 KiB |
| Scoring plan (before extraction) | `ee3c2b40d01c8bd6eefb6e11f7f76484302ed06b2d3812db93d8f7acf969818b` |
| Scoring PRE | `25d4f5dad44f29d367f32db5bf627dba421c4cda6f65edf488b5a7ff44657009` |
| Scoring job | outer0, cleanup known, 0.0935 s, peak sampled2224 KiB |
| Six-view neutral/marked render jobs | both outer0/cleanup known; 6.6662/6.7103 s |
| Independent score/endpoint POST | `9bdc50c7d77ce288b2eaf7ef977602f83e423038b39cf5481394a9ddc3faa03c` |

Final independent counter, without importing the scorer, reproduced12TP/0FP/
0FN. All12 rendered endpoint pairs match extraction exactly; image hashes and
all terminal cleanup/budget records passed. Final cumulative case outputs
6,866,458 bytes, within40 MB. The independent reviewer checked custody and
endpoint bindings; pixel inspection was performed by root, not fabricated as
a second visual review.

Candidate/neutral/extraction commands use
`scripts/evaluate_cad_email_transfer_v2.py --attempt <attempt> --mode
neutral|extraction --frozen-sha256 <pin> --pre <mode PRE> --pre-sha256 <pin>`;
extraction additionally binds exact neutral POST and reference paths/hashes.
The scorer command and all exact argument pins are retained verbatim in
`score/EXECUTION.json`. Renderer exact commands/source pins are retained in
`NEUTRAL.supervision/FROZEN.json` and `PREDICTIONS.supervision/FROZEN.json`.
These are terminal audit records, not instructions to rerun existing attempts.
A justified successor requires a new frozen attempt and independent gates.

V1 preparation remains a retained negative before native launch: its full
runtime archive exceeded the initial aggregate output cap. V2 separates a
fixed300 MB pinned runtime archive from40 MB cumulative case outputs; unchanged
60 s/4 GiB native caps. See [V2 protocol](cad-email-transfer-v2-2026-10-01.md).

## Visual review and limits

Root inspected all six neutral and all six marked views. Cyan marker segments
follow the projected bend regions, including short tabs; hole walls have no
cyan bend marks. Projection labels are crowded and line overlays are not a
hidden-surface test. Marker endpoints are exported exactly to the image
manifest for independent verification. Displays do not prove continuous trim
geometry, full-width manufacturing centerlines, dimensional accuracy or units.
Those dimensions remain unscored. No unfolding was executed in this transfer.
Public UI/runtime defaults are unchanged.

The newest email CAD is native SLDPRT without a STEP attachment, unsupported by
the current controlled importer. Do not describe this STEP success as native
SolidWorks or CAD-agnostic format support. Separately, the case17 derived-only
diagnostic explains two remaining development misses as strict negative-join
predicate refusals at2–4e-13 units. A safe local axial-component successor is
justified for investigation, but no tolerance relaxation or engine patch has
been promoted on that evidence alone.
