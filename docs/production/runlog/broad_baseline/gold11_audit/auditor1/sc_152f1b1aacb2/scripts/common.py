import numpy as np, json
S = "/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/"
OUT = "/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor1/sc_152f1b1aacb2/"
P = np.load(S+"P.npy"); R=np.load(S+"R.npy"); c=np.load(S+"c.npy")
gold = json.load(open("/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/bend_line_gold_catalog_v1.json"))
case = [cc for cc in gold['cases'] if cc['case_id']=='corpus/08_40500000'][0]
G = {b['bend_id']:np.array([(np.array(b['line_start'])-c)@R.T,(np.array(b['line_end'])-c)@R.T]) for b in case['bends']}
def gold_at(L, axis, val):
    """point on gold segment L (2x3, part frame) where coordinate 'axis' == val; returns (pt, inside_span)"""
    a,b = L
    if abs(b[axis]-a[axis])<1e-9: return None, False
    t = (val-a[axis])/(b[axis]-a[axis])
    return a + t*(b-a), (0<=t<=1)
def to_raw(Xp):
    return Xp@R + c
