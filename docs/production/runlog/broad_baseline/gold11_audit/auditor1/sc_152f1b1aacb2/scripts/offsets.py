import numpy as np, json
exec(open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/apexlines.py").read().split("folds = {")[0])
def seg_dist(X, L):
    a,b=L; d=b-a; Ld=np.linalg.norm(d); d/=Ld
    t=(X-a)@d; foot=a+np.outer(t,d); return np.linalg.norm(X-foot,axis=1), t, Ld
out={}
for gid,kind,sv in (("M01","B1",np.arange(-63.0,66.01,1.0)),("M02","B2",np.arange(-63.0,66.51,1.0)),("M07","B3",np.arange(-24.5,26.51,1.0))):
    X,A=sweep(kind,sv)
    dist,t,Ld=seg_dist(X,G[gid])
    inside=(t>=0)&(t<=Ld)
    out[gid]=dict(n=int(len(X)), offset_median=float(np.median(dist)), offset_p90=float(np.percentile(dist,90)), offset_max=float(dist.max()), apex_station_e1_range=[float(X[:,0].min()),float(X[:,0].max())], frac_apex_stations_within_gold_span=float(inside.mean()))
    print(gid, {k:(round(v,3) if isinstance(v,float) else v) for k,v in out[gid].items()})
json.dump(out,open(OUT+"a1_gold_apex_offsets.json","w"),indent=1)
