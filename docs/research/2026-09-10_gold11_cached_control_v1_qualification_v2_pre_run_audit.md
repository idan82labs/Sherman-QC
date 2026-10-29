# Cached-control V1 qualification V2: exact pre-run audit bundle

Date:2026-09-10. Prospective freeze; execution awaits independent GREEN.
See [qualification V2 protocol](2026-09-10_gold11_cached_control_v1_qualification_v2_protocol.md).
The predecessor is terminal FAILED and its smoke remains unexecuted. Candidate
runtime is unchanged; qualification successor changes only its test fixture/opt-in,
adds five checks and narrowly additive harness routes. Root inspected the entire
new-test diff against the previously fully read predecessor and the complete
harness diff against archived e11d1343. No tests/data run during this freeze.

Expected source141PASS/one declared smoke skip, then conditional smoke142PASS;
unchanged60wall/45CPU/1GiB/1MiB and8wall/5CPU/64KiB child budgets. No common-file
edit between gates. Independent exact pre/post audits required, no retries.

| Source | SHA256 |
| --- | --- |
| `scripts/run_gold11_cached_control_v1.py` | `75af29ad6314c63a98ae5176d348a82e55fd408e975f92766879cfb09939721f` |
| `scripts/run_gold11_mesh_development_v2.py` | `94d51b90faef0d88b15ed2d3d84d5b7d36ba94bfb6f63f90b9cc555a35e26422` |
| `tests/test_gold11_cached_control_v1.py` | `67ec0c5dccebfd58298818e6c43df2651eeee2492c636427f15d4776241dd5e6` |
| `tests/test_gold11_cached_control_v1_qualification_v2.py` | `2b9f1339b76ecbb5813f575ec54849477532bd0c1b89bb2ca7fc4d722b47934f` |
| `docs/research/2026-09-10_gold11_cached_control_v1_protocol.md` | `7f6dc2894544bbe2adf4ce45ca73a521bed7f7407a431f8d097c223448d54d37` |
| `docs/research/2026-09-10_gold11_cached_control_v1_qualification_protocol.md` | `60a4edc4f2ab914e4dc59c3d39c9dafaba1aa675944beb99446b09e8fc49e362` |
| `docs/research/2026-09-10_gold11_cached_control_v1_qualification_v2_protocol.md` | `cb5c100558f7c35e1e47d5facebf86ce5e10fc9d8c3fc1e0ab83d811c0988caf` |
| `scripts/check_cad_guided_measurement.py` | `0fb68e412bd0fb52b071f11d28c31e15849592a970025e47635321109c8ed14f` |

The actual cached-run16-file bundle retains the old failed test as historical
evidence. Its future PRE_RUN_AUDIT must bind this successor closure and its two
terminal receipts/OUTPUT hashes; neither this freeze nor an agent statement is
actual cached-data execution authorization before those gates pass.
