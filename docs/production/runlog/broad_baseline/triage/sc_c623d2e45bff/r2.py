import numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/c623/'
Y=np.load(S+'Y.npy')
fig,axs=plt.subplots(2,3,figsize=(24,14))
# 1: elev section through middle of plate width (e2 in [-5,5]) e1 vs e3
m=np.abs(Y[:,1])<5; axs[0,0].scatter(Y[m,0],Y[m,2],s=1); axs[0,0].set_title('e1-e3 slice |e2|<5'); axs[0,0].set_aspect('equal')
# 2: end region e1<-70, plan e2-e3? show e1 vs e3 near e2 = 50
m=(Y[:,0]<-60)&(np.abs(Y[:,1]-50)<5); axs[0,1].scatter(Y[m,0],Y[m,2],s=2,c=Y[m,1]); axs[0,1].set_title('corner +e2: e1-e3, 45<e2<55'); axs[0,1].set_aspect('equal')
m=(Y[:,0]<-60); axs[0,2].scatter(Y[m,1],Y[m,2],s=1,c=Y[m,0]); axs[0,2].set_title('end e1<-60: e2-e3'); axs[0,2].set_aspect('equal')
# section e2 near +52 wall region, e1 beyond walls (e1 in 55..90): e2-e3
m=(Y[:,0]>55); axs[1,0].scatter(Y[m,1],Y[m,2],s=1,c=Y[m,0]); axs[1,0].set_title('e1>55 (beyond walls): e2-e3'); axs[1,0].set_aspect('equal')
m=(Y[:,0]<-50)&(Y[:,0]>-80); axs[1,1].scatter(Y[m,1],Y[m,2],s=1,c=Y[m,0]); axs[1,1].set_title('-80<e1<-50: e2-e3'); axs[1,1].set_aspect('equal')
# plan close-up corner
m=(Y[:,0]<-40)&(Y[:,1]>35); sc=axs[1,2].scatter(Y[m,0],Y[m,1],s=1,c=Y[m,2]); plt.colorbar(sc,ax=axs[1,2]); axs[1,2].set_title('plan corner (+e2, -e1) colour=e3'); axs[1,2].set_aspect('equal')
plt.savefig(S+'sec.png',dpi=70)
