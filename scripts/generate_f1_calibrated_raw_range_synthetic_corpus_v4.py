#!/usr/bin/env python3
"""Generate prospective calibrated V4 with 15 new complex profile families.

V4 reuses the frozen scanner renderer but replaces every V3 cross-section.
Its private truth remains sealed until V61/V62 inference and visual audit are
frozen.  It is synthetic engineering evidence, not physical confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.generate_f1_calibrated_raw_range_synthetic_corpus_v3 as base


TOKEN = "SYNTHETIC_CALIBRATED_RAW_RANGE_V4_PROSPECTIVE_COMPLEX_NON_CONFIRMATORY"
CORPUS_KIND = "sherman_qc_f1_calibrated_raw_range_prospective_complex_v4"
CAPTURE_INDICES = (8, 9)


FAMILIES: tuple[dict[str, Any], ...] = (
    {"id":"PF01","name":"triple_micro_crown","profile":[(0,0),(16,0),(19,2.4),(24,2.4),(27,0),(39,0),(43,3.2),(49,3.2),(53,0),(66,0),(71,5.5),(79,5.5),(84,0),(104,0)],"length":176,"radius":0.65,"challenge":"three unequal close micro-beads from 2.4 to 5.5 mm"},
    {"id":"PF02","name":"rolled_zed_return","profile":[(0,0),(18,0),(27,9),(31,34),(43,48),(70,48),(78,39),(82,14),(91,5),(116,5)],"length":226,"radius":7.0,"challenge":"asymmetric Z transition with broad rolled shoulders and short return"},
    {"id":"PF03","name":"multi_radius_omega","profile":[(0,0),(11,0),(17,7),(20,29),(28,43),(43,52),(69,52),(83,43),(91,27),(94,7),(101,0),(121,0)],"length":232,"radius":5.5,"challenge":"many-facet omega approximating two different shoulder radii"},
    {"id":"PF04","name":"inverted_nested_hat","profile":[(8,54),(0,54),(0,12),(5,6),(14,6),(20,15),(67,15),(73,6),(82,6),(87,12),(87,54),(78,54),(73,49)],"length":214,"radius":1.8,"challenge":"deep inverted hat with nested shelves and terminal hooks"},
    {"id":"PF05","name":"faceted_low_crown","profile":[(0,0),(20,0),(27,1.8),(35,3.4),(44,4.7),(54,5.3),(64,4.7),(73,3.4),(81,1.8),(88,0),(112,0)],"length":204,"radius":6.5,"challenge":"very shallow crown distributed across eight sub-15-degree turns"},
    {"id":"PF06","name":"staggered_shelf_joggle","profile":[(0,0),(17,0),(22,4),(22,12),(29,18),(42,18),(47,23),(61,23),(67,17),(67,9),(75,2),(101,2)],"length":184,"radius":1.25,"challenge":"two staggered shelves with unequal close joggles"},
    {"id":"PF07","name":"broad_bulb_channel","profile":[(0,35),(3,19),(10,7),(21,1),(78,1),(90,7),(98,20),(101,36),(96,48),(87,55)],"length":238,"radius":10.0,"challenge":"broad rounded channel ending in an asymmetric bulb return"},
    {"id":"PF08","name":"subthree_step_ladder","profile":[(0,0),(14,0),(14,2.0),(20,2.0),(20,4.6),(27,4.6),(27,7.7),(35,7.7),(35,11.2),(47,11.2),(53,16),(82,16)],"length":158,"radius":0.5,"challenge":"2.0 to 3.5 mm step ladder at scanner sampling limits"},
    {"id":"PF09","name":"asymmetric_w_return","profile":[(0,0),(13,0),(18,6),(25,25),(36,36),(51,27),(65,39),(79,34),(86,17),(93,7),(119,7)],"length":244,"radius":3.8,"challenge":"asymmetric W with alternating acute and rounded turns"},
    {"id":"PF10","name":"rolled_box_lip","profile":[(13,61),(5,61),(0,54),(0,15),(8,4),(25,0),(72,0),(88,5),(96,17),(96,52),(91,59),(80,59),(75,54)],"length":254,"radius":8.5,"challenge":"large rolled box with short double lip on one wall"},
    {"id":"PF11","name":"paired_micro_bead_shelf","profile":[(0,0),(19,0),(22,2.3),(27,2.3),(30,0),(43,0),(46,2.8),(52,2.8),(55,0),(70,0),(76,7),(99,7)],"length":166,"radius":0.6,"challenge":"two close low-amplitude beads beside a larger shelf"},
    {"id":"PF12","name":"deep_multifacet_trough","profile":[(0,31),(4,18),(11,8),(22,2),(38,0),(77,0),(93,3),(104,10),(111,21),(114,35)],"length":218,"radius":9.5,"challenge":"deep trough whose sides distribute curvature over four facets"},
    {"id":"PF13","name":"alternating_micro_macro_ribs","profile":[(0,0),(12,0),(15,2.5),(20,2.5),(23,0),(36,0),(44,13),(57,13),(63,4),(69,4),(72,0),(87,0),(92,6),(101,6),(106,0),(126,0)],"length":194,"radius":0.85,"challenge":"alternating micro and macro ribs with irregular pitch"},
    {"id":"PF14","name":"hooked_hat_double_return","profile":[(6,57),(0,57),(0,9),(5,3),(15,3),(21,10),(28,49),(38,60),(72,60),(82,49),(89,10),(95,3),(105,3),(111,9),(111,57),(104,57),(99,52)],"length":228,"radius":2.8,"challenge":"deep hat with nested return hooks on both sides"},
    {"id":"PF15","name":"long_mixed_radius_labyrinth","profile":[(0,0),(13,0),(17,3),(21,12),(29,20),(41,20),(47,27),(60,34),(75,34),(83,29),(89,18),(93,8),(101,3),(117,3),(123,9),(142,9)],"length":268,"radius":2.3,"challenge":"fifteen-turn labyrinth mixing short creases and broad transitions"},
)


def generate(output_root: Path, *, width: int = base.capture.DEFAULT_WIDTH, height: int = base.capture.DEFAULT_HEIGHT, view_count: int = base.capture.DEFAULT_VIEW_COUNT, family_limit: int | None = None) -> dict[str, Any]:
    previous = (base.TOKEN, base.CORPUS_KIND, base.CAPTURE_INDICES, base.FAMILIES)
    base.TOKEN, base.CORPUS_KIND, base.CAPTURE_INDICES, base.FAMILIES = TOKEN, CORPUS_KIND, CAPTURE_INDICES, FAMILIES
    try:
        report = base.generate(output_root, width=width, height=height, view_count=view_count, family_limit=family_limit)
    finally:
        base.TOKEN, base.CORPUS_KIND, base.CAPTURE_INDICES, base.FAMILIES = previous
    renderer_path = output_root / "evaluator_only" / "renderer_manifest.json"
    renderer = json.loads(renderer_path.read_text(encoding="utf-8"))
    renderer["generator_sha256"] = base.capture._sha256(Path(__file__).resolve())
    renderer["parent_v3_generator_sha256"] = base.capture._sha256(Path(base.__file__).resolve())
    base.capture._atomic_json(renderer_path, renderer)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=base.capture.DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=base.capture.DEFAULT_HEIGHT)
    parser.add_argument("--view-count", type=int, default=base.capture.DEFAULT_VIEW_COUNT)
    parser.add_argument("--family-limit", type=int)
    args = parser.parse_args()
    report = generate(args.output_root.resolve(), width=args.width, height=args.height, view_count=args.view_count, family_limit=args.family_limit)
    print(json.dumps({"output_root":str(args.output_root.resolve()),"case_count":report["case_count"],"family_count":report["family_count"],"confirmatory_eligible":False,"token":TOKEN}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
