import json,glob,sys
for d in sorted(glob.glob('runs/N_*')):
  try: r=json.load(open(d+'/record.json'))
  except: continue
  n=d.split('/')[-1][2:]
  if not r.get('e3m'): print(n.ljust(18), r.get('state'), r.get('summary'), 'NO e3m', r.get('reasons')); continue
  for p in r['e3m'][:3]:
    f=lambda x: 'NA' if x is None else round(x,3)
    if 'mirror_total' not in p: print(n.ljust(18), r['state'][:12], 'no mirror', p.get('sub')); continue
    pp=p.get('proper_pool') or []; mp=p.get('mirror_pool') or []
    hp=max([a[2]-a[3] for a in pp],default=None); hm=max([a[2]-a[3] for a in mp],default=None)
    print(n.ljust(18), r['state'][:12].ljust(12), 'pass',str(r['n_pass']).ljust(3), 'sub', 'P' if p['sub'] is None else int(p['sub'][1]), 'susp',int(p['suspicious']), 'expl',f(p['best_explained']),'->',f(p['mirror_explained']),'dt',f(p['mirror_total']-p['best_total']),'de',f(p['mirror_explained']-p['best_explained']),'holes',hp,hm,'sym',p['mirror_selfmaps_decoded'],'rms',f(p.get('best_stage_min_mirror_rms')),'ms',p['mirror_s'], 'w', r['wall_s'])
