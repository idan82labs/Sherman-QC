import json,sys,os
E3=sys.argv[1]
spec=sys.argv[2:]  # case:B1,B4
for sp in spec:
    c,labs=sp.split(':')
    labs=labs.split(',')
    r=json.load(open(f"{E3}/{c}/engine/result.json"))
    for b in r['bends']:
        if b['label'] not in labs: continue
        print("==",c,b['label'],b['state'], b.get('state_reason') or '')
        for ch in b['characteristics']:
            ev=ch.get('evidence') or {}
            print(f"   {ch['kind']:18s} {ch.get('side') or '':2s} {ch.get('boundary_kind') or '':14s} exp={ch['expected']} act={ch['actual']} U={ch['uncertainty']} {ch['decision']:8s} {ch.get('reason') or ''} | {ev.get('method','')} | {';'.join(ev.get('notes',[]))[:200]}")
