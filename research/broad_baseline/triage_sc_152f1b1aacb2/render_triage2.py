import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('research/broad_baseline/triage_sc_152f1b1aacb2/render_triage.py').read().split('fig,axs')[0])
fig,axs=plt.subplots(4,1,figsize=(16,20))
m=q[:,1]<-80; draw(axs[0],0,2,m); axs[0].set_title('wall face e2<-80 (M01/M05/M06/M07 wall), e1-e3')
m=q[:,1]>80; draw(axs[1],0,2,m); axs[1].set_title('wall face e2>80 (M02/M03/M04 wall), e1-e3')
m=np.abs(q[:,1])<40; draw(axs[2],0,2,m); axs[2].set_title('side section |e2|<40, e1-e3 (short edges at e1=+-105)')
axs[2].set_ylim(-12,12)
m=q[:,2]>-2; draw(axs[3],0,1,m); axs[3].set_title('top view, only points with e3>-2 (above base plate)')
plt.tight_layout(); plt.savefig(OUT+'views_walls.png',dpi=80)
# close-ups of wall corners
fig,axs=plt.subplots(2,4,figsize=(20,10))
for col,(k,e2s) in enumerate([('M05',-1),('M06',-1),('M04',1),('M03',1)]):
    a,b=G[k]; A=L(a)[0]; x0=A[0]
    m=(np.abs(q[:,0]-x0)<20)&(q[:,1]*e2s>60)&(q[:,2]>-2)&(q[:,2]<45)
    draw(axs[0,col],0,1,m,lim=((x0-20,x0+20),(e2s*95-25,e2s*95+25) if e2s>0 else (-120,-70)))
    axs[0,col].set_title(f'{k} close-up top view (e3 in -2..45)')
    m2=(np.abs(q[:,0]-x0)<20)&(q[:,1]*e2s>70)
    draw(axs[1,col],0,2,m2,lim=((x0-20,x0+20),(-10,45)))
    axs[1,col].set_title(f'{k} close-up wall face e1-e3')
plt.tight_layout(); plt.savefig(OUT+'closeups_M03_M06.png',dpi=80)
