"""Frozen identity table of the Gold11 development cases (consumed, authorized).

Sources: docs/research/2026-09-10_gold11_a0_provenance_trace_v7.md (A0 tokens and
hashes), docs/research/2026-09-10_gold11_retained_control_recovery.md (CAD files
and hashes), data/bend_corpus/annotations/bend_line_gold_catalog_v1.json (gold;
SHA256 4ad243c1...b36, selected 11-case snapshot d16390b2...45dc).

Historical headline (cached control V2, 2026-09-10): UNION 30 TP / 30 FP / 41 FN,
precision 0.5, recall 30/71, line-F1 0.4580152671755725.
"""

from __future__ import annotations

CATALOG_SHA256 = "4ad243c1f8d7332749c854f32729607f156cd2e0f47454acd4b0ff61f1793b36"
SELECTED_SNAPSHOT_SHA256 = "d16390b2c05157874444f4c3507558bddf28e93b54ccb148228f75248c2b45dc"
HISTORICAL_UNION_F1 = 0.4580152671755725
HISTORICAL = {
    "scan_only_a0": {"lines": 33, "tp": 19, "fp": 14, "fn": 52, "f1": 0.36538461538461536},
    "cad_only_gated": {"lines": 44, "tp": 20, "fp": 24, "fn": 51, "f1": 0.34782608695652173},
    "intersection": {"lines": 17, "tp": 10, "fp": 7, "fn": 61, "f1": 0.22727272727272727},
    "union": {"lines": 60, "tp": 30, "fp": 30, "fn": 41, "f1": 0.4580152671755725},
}

A0_ROOT = "output/research/f1-gold27-20260803/structured_union_exact_v1/cases"   # <token>/prediction_owned_regions.json
CAD_ROOT = "output/research/f1-cad-guided-gold13-e2e-20260822/predictions_scan_frame_gated"  # <case>/semantic_result.json

# case_id: (family, catalog case, A0 token, scan sha256, retained A0 sha256, retained gated-CAD sha256)
CASES = {
    "CG08506AE97D": ("10839000", "corpus/03_10839000", "C003",
                     "cec08dfca9ab309ff3eef42cd3dc72eb33168e1d43029f3a0658f2923e4e171b",
                     "df42a588cf1747e358c9f8fd8704a5b3c7d01388e5b85cc6df271edf2680d060",
                     "51d9238df430c2090eb5d5787e24dcdc90aa299dbcb1bf279b59d18d332a014e"),
    "CG191DEC7F81": ("49001000", "corpus/10_49001000", "C021",
                     "7a1773eca9fab616afef498a9f9831e67eacb17fcc0d170607490a9e9db37020",
                     "5c0ef9e5743e854550e40dea7808ec11570b897a076dd883dab67e1ae1f9b06a",
                     "ecd34f506c1510d42e0d4bc4f6e540613c9477dfe307b221a36d7fdae2f6631d"),
    "CG1E0BBA2933": ("38172000", "shaked_xv/C05", "C011",
                     "4f8188b4d1a844305134801c3bb5364270250416d2f4dc43f0b05d37cc7962b9",
                     "2eac14dd768471a540460f60ac8a696ae04a7d0abfcb4880618bfd4965fb5353",
                     "3d42db809281eeca6b8d1c5ae0f8c97ec2aea8f7f2d55c578ad058778521e6e8"),
    "CG4BDD1371B6": ("37361005", "corpus/05_37361005", "C008",
                     "d1732de180230bdbf0caa6eaf454a33071c206060aea276dce67a9abcf13b038",
                     "bdefea6cfab50cec78907cc997b1ad28b1ac8755a1b448444017f7a6f6216d4a",
                     "a2c5f4b000f65932c44b238414fd428ca0b2f9f9f31cf5e98e6fa238c48fcc3e"),
    "CG4FAF273093": ("49024000", "corpus/11_49024000", "C022",
                     "70b39f530f00a72dd99acf00b4a065f45bfe4f50185e3579c3ba26143f7d2ade",
                     "af8ce6e7827b1adb9fea871ceb5406d28e2871301a145268290eca3708b2dcc9",
                     "2fb45cc5366c9527b8f84fd09ab90beae70978b653c6cea6f8354046ef96be3b"),
    "CG64C412CC73": ("40500000", "corpus/08_40500000", "C015",
                     "152f1b1aacb20920a601f7a8197d4c89a111688325bcf557a6a731641f306614",
                     "f464aa39f3a16d1dd6272fc08a735b38c215ca8b7ad5b32dcdfe93101db61176",
                     "7fd5fd125403af4f965f22a61cbe7781290a6e1326763eea47d74ff88d4172cd"),
    "CG65CD575B1C": ("38172000", "corpus/06_38172000", "C009",
                     "07062eee17aa4e41721b93cdb69ddb4b752e6713a16c5aab0a10df9c86d07880",
                     "6160c39dee6998208c9f333855e9e402ed486282dddee7494b43a0104b0837a7",
                     "dfd6e66b5716a9c2ecb1b12798a8a1b2ae6b93e1fdeba3b7bfbad6502146464d"),
    "CG84C8FA12D2": ("10839000", "corpus/01_10839000", "C001",
                     "71cf0b216f43284e401c891091cd56c10321aa3eebccde204f8d206683999215",
                     "3befcc1d963ce9ab70a915e174c3be5503c5a81a46fa16c9f8883c23db9b7914",
                     "685a2b0ddc86af01e4179f450e7bf45a5ed471267db7e04cae01364c2785d966"),
    "CGE4B0549E41": ("10839000", "corpus/02_10839000", "C002",
                     "08686dc1d5d4c89725393907082e8cc4e4c0f5a21427697760ff44d29c7939fc",
                     "eb0b050d970b48108c662f8e943002bfdaf6c3a01a978b4483ca1a6d0c83015f",
                     "27713f575b55867de7d73cdfbba960a65bc40d25c2e51179831b02daab9066ad"),
    "CGEBA9A869AC": ("38172000", "corpus/07_38172000", "C010",
                     "f2475286312b7e11be773166afae5d7eaacd826f8b59cee84289d05b4a222b63",
                     "9e602684ecd345ace1a840743f0412d14d9106629c1a760dd1b0133d83fa8299",
                     "22209579f6ee4a965d0b6b96b6140af9e6e68aade335e9c43bce95de54618231"),
    "CGF41F49066A": ("38172000", "shaked_xv/C06", "C012",
                     "2714fb7e0c0b681196da62bf368e60c0635b97000f34df3241ca1b70ee56faba",
                     "02b0e6447e688951294e56a672ff42271ca74ebe8232345e386917aa2ed3b612",
                     "8f6286463ca37c7ebc76b124ab7d48c63faa53c411295a66374c205c612b5d34"),
}

# CAD files the blind CAD lane used (output/research/f1-cad-guided-gold13-e2e-20260822/summary.json).
CAD_BY_FAMILY = {
    "10839000": "~/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01.stp",
    "37361005": "~/ShermanQC/source-assets/02_CAD_QC_3D/37361005_01.stp",
    "38172000": "~/ShermanQC/source-assets/02_CAD_QC_3D/38172000_01.stp",
    "40500000": "~/ShermanQC/source-assets/02_CAD_QC_3D/40500000_0 1.stp",
    "49001000": "~/ShermanQC/source-assets/02_CAD_QC_3D/49001000_A-PRELIM.stp",
    "49024000": "~/ShermanQC/source-assets/02_CAD_QC_3D/49024000_A.stp",
}


def family(case_id: str) -> str:
    return CASES[case_id][0]


def catalog_case(case_id: str) -> str:
    return CASES[case_id][1]


def token(case_id: str) -> str:
    return CASES[case_id][2]


def scan_sha(case_id: str) -> str:
    return CASES[case_id][3]


def a0_sha(case_id: str) -> str:
    return CASES[case_id][4]


def cad_sha(case_id: str) -> str:
    return CASES[case_id][5]
