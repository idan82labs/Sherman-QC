import json, sys
d = json.load(open(sys.argv[1]))
reg = (d.get('scan') or {}).get('registration') or {}
print('warnings', [w[:70] for w in d.get('warnings', [])])
print('reg', {k: reg.get(k) for k in ('in_plane_degenerate', 'pose_ambiguous', 'rmse_mm')}, 'summary', d['summary'].get('state'))
for b in d['bends']:
    out = []
    for ch in b['characteristics']:
        k = ch['kind'][:4] + (ch.get('side') or '')
        out.append(f"{k}:{ch['state'][:4]}/{ch['decision'][:3]}/{ch.get('reason') or ''}/{'' if ch['deviation'] is None else round(ch['deviation'], 3)}"
                   f"±{'' if ch['uncertainty'] is None else round(ch['uncertainty'], 3)}")
    print(b['label'], b['state'], ' '.join(out))
