import json, sys
S = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1i4"
W = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_iter4"
for name in sys.argv[1:]:
    if name == "prev":
        an, sc = W + "/phys/analysis", W + "/phys/physical_scorecard.json"
    else:
        an, sc = f"{S}/phys/analysis_{name}", f"{S}/phys/physical_scorecard_{name}.json"
    g = json.load(open(f"{an}/goldset_scores.json")); d = json.load(open(f"{an}/subset_scores.json")); s = json.load(open(sc))
    b = d["BROAD_all26"]["engine"]; g2 = s["G2"]; g3 = s["G3"]
    rl = g2["line_metric_dual_report"]["relabel"]
    print(f"{name:5s} GOLD_SET micro {g['e3']['micro_f1']:.3f} macro {g['e3']['family_macro_f1']:.3f} (tp {g['e3']['tp']} fp {g['e3']['fp']} fn {g['e3']['fn']}) | BROAD micro {b['micro']['f1']:.3f} macro {b['family_macro_f1']:.3f} (tp {b['true_positive']} fp {b['false_positive']} fn {b['false_negative']}) | relabel F1 {rl['f1']:.3f} P {rl['precision']:.3f} R {rl['recall']:.3f}")
    cov = g2["coverage"]; ak = g2["angle_known_good"]; fk = g2["flange_known_good"]; rp = g2["repeatability"]; rt = g2["runtime"]
    print(f"       G2 gates {g2['gates']}")
    print(f"       b angle {cov['angle']['measured']}/{cov['angle']['planned']} flange {cov['flange_sides']['measured']}/{cov['flange_sides']['planned']} | c MAE {ak['e3_mae_deg']:.3f} ({ak['e3_measured']}) | d P95 {fk['p95_abs_mm']:.3f} max {fk['max_abs_mm']:.3f} | f angle P95 {rp['e3_p95_angle_deg']:.3f} flange P95 {rp['e3_p95_flange_mm']:.3f} | e P95 {rt['p95_wall_s']:.1f} max {rt['max_wall_s']:.1f}")
    print(f"       G3 match {g3['state_match_rate']:.3f} ({g3['matches']}/{g3['bends_scored']}) gates {g3['gates']}")
    print(f"       G3 formed angle cov {g3['formed_bends_coverage']['angle']['measured']}/{g3['formed_bends_coverage']['angle']['planned']} flange {g3['formed_bends_coverage']['flange_sides']['measured']}/{g3['formed_bends_coverage']['flange_sides']['planned']} | known-good angle MAE {g3['formed_bends_known_good_angle']['e3_mae_deg']:.3f} | flange P95 {g3['formed_bends_known_good_flange']['p95_abs_mm']:.3f}")
