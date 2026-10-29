import json, numpy as np
GOLD='/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/bend_line_gold_catalog_v1.json'
def load_scan():
    d=np.load('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a1_f247/scan.npz')
    return d['V'],d['F'],d['N']
def load_gold():
    d=json.load(open(GOLD))
    c=[c for c in d['cases'] if c['case_id']=='corpus/07_38172000'][0]
    out={}
    for b in c['bends']:
        out[b['bend_id']]=(np.array(b['line_start'],float),np.array(b['line_end'],float),b)
    return out,c
def unit(v):
    v=np.asarray(v,float); return v/np.linalg.norm(v)
