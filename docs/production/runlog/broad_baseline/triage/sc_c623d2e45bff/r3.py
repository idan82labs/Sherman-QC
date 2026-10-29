import numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/c623/'
Y=np.load(S+'Y.npy')
for sgn,name in ((1,'B1 (+e2)'),(-1,'B2 (-e2)')):
    m=(np.abs(Y[:,1]-sgn*52)<4)&(np.abs(Y[:,2]+8)<4)
    h,_=np.histogram(Y[m,0],bins=np.arange(-90,95,5)); print(name,'bend-zone pts per 5mm e1 bin from -90:',h.tolist())
    m2=(np.abs(Y[:,1]-sgn*52)<2)&(Y[:,2]>0)&(Y[:,2]<20)
    h2,_=np.histogram(Y[m2,0],bins=np.arange(-90,95,5)); print(name,'wall pts per bin:',h2.tolist())
fig,axs=plt.subplots(2,1,figsize=(18,8))
for ax,sgn in zip(axs,(1,-1)):
    m=(np.abs(Y[:,1]-sgn*52)<6)&(Y[:,2]<5)
    sc=ax.scatter(Y[m,0],Y[m,1],s=1,c=Y[m,2]); ax.set_aspect('equal'); plt.colorbar(sc,ax=ax)
    ax.axvline(-48);ax.axvline(49.3);ax.axvline(32.4,color='r')
    ax.set_title(f'plate-wall bend zone e2~{sgn*52} plan, colour=e3 (red line = gold B2 end 32.4)')
plt.savefig(S+'bz.png',dpi=70)
