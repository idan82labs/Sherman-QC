import numpy as np
exec(open("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/aud1_sc152f1b_private/common.py").read())
N=np.load(S+"N.npy"); Np=N@R.T; Np/=np.linalg.norm(Np,axis=1,keepdims=True)+1e-12
for ax_i,name in ((2,"horizontal faces (normal ~ e3): e3 levels"),(1,"faces with normal ~ e2: e2 levels"),(0,"faces with normal ~ e1: e1 levels")):
    m=np.abs(Np[:,ax_i])>0.985
    vals=P[m,ax_i]
    h,ed=np.histogram(vals,bins=np.arange(np.floor(vals.min())-0.25,np.ceil(vals.max())+0.5,0.5))
    pk=[(round((ed[i]+ed[i+1])/2,2),int(h[i])) for i in range(len(h)) if h[i]>=150]
    print(name, "n=",m.sum()); print("   ", pk)
