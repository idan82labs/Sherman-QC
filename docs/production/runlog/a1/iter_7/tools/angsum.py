import json,sys,os
E3=sys.argv[1]
cases=sys.argv[2:]
for c in cases:
    p=f"{E3}/{c}/engine/result.json"
    if not os.path.exists(p):
        print(c,"missing");continue
    r=json.load(open(p))
    print("=====",c, r.get('part_name'), r['summary'].get('overall') if isinstance(r.get('summary'),dict) else '')
    for b in r['bends']:
        lab=b.get('label') or b.get('bend_label')
        st=b.get('state')
        for ch in b['characteristics']:
            if ch['kind']!='INCLUDED_ANGLE': continue
            notes=';'.join(n for n in (ch.get('evidence') or {}).get('notes',[]) )
            s=ch.get('stations') or {}
            print(f"  {lab:4s} {st:18s} nom={ch['expected']} act={ch['actual'] and round(ch['actual'],3)} dev={ch['deviation'] and round(ch['deviation'],3)} U={ch['uncertainty'] and round(ch['uncertainty'],3)} {ch['decision']:7s} {ch.get('reason') or ''} {(ch.get('evidence') or {}).get('method','')} st[{s.get('min') and round(s['min'],2)},{s.get('max') and round(s['max'],2)}]n={s.get('count')} {notes}")
