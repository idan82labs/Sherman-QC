import sys, numpy as np, json, math
sys.path.insert(0,'/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714')
import xs
F=np.load(xs.W+'frame.npy'); o=F[0]; R=F[1:]
def raw(p): return o+np.asarray(p)@R
cands={'SF_L':(raw([-112.6,74.0,30.0]),raw([-112.6,117.0,30.0])),
       'SF_R':(raw([111.4,74.0,30.6]),raw([111.4,117.0,30.6])),
       'TAB09_main':(xs.GOLD['B09'][0],xs.GOLD['B09'][1]),
       'TAB10_main':(xs.GOLD['B10'][0],xs.GOLD['B10'][1])}
for k,(a,b) in cands.items(): print(k,(a-o)@R.T,(b-o)@R.T)
xs.GOLD.update({k:v for k,v in cands.items() if k.startswith('SF')})
