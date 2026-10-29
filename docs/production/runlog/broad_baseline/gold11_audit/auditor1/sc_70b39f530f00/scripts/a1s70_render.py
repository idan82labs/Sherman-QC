import numpy as np, trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def unit(v):
    v=np.asarray(v,float); return v/np.linalg.norm(v)

def basis(view, up):
    d=unit(view)            # direction from scene towards camera
    up=np.asarray(up,float)
    r=unit(np.cross(up,d))
    u=np.cross(d,r)
    return r,u,d

class Splat:
    def __init__(self, mesh, n=1500000, seed=0):
        pts, fid = trimesh.sample.sample_surface(mesh, n, seed=seed)
        self.p=pts; self.n=mesh.face_normals[fid]; self.fid=fid
    def render(self, view, up, res=0.1, pad=3.0, center=None, half=None, color=None, light=None, clip=None):
        r,u,d=basis(view,up)
        P=self.p; N=self.n
        if clip is not None:
            keep=clip(P); P=P[keep]; N=N[keep]
        x=P@r; y=P@u; z=P@d
        if center is None:
            x0,x1=x.min()-pad,x.max()+pad; y0,y1=y.min()-pad,y.max()+pad
        else:
            cx,cy=np.dot(center,r),np.dot(center,u)
            hx,hy=half if np.ndim(half) else (half,half)
            x0,x1,y0,y1=cx-hx,cx+hx,cy-hy,cy+hy
        W=int(np.ceil((x1-x0)/res)); H=int(np.ceil((y1-y0)/res))
        ix=((x-x0)/res).astype(int); iy=((y1-y)/res).astype(int)
        ok=(ix>=0)&(ix<W)&(iy>=0)&(iy<H)
        ix,iy,z,NN=ix[ok],iy[ok],z[ok],N[ok]
        pix=iy*W+ix
        order=np.lexsort((z,pix))
        pix_s=pix[order]
        last=np.r_[pix_s[1:]!=pix_s[:-1],True]
        sel=order[last]
        img=np.ones((H*W,3))
        L=unit(light) if light is not None else d
        sh=0.25+0.75*np.abs(NN[sel]@L)
        if color is None:
            col=np.stack([sh,sh,sh],1)
        else:
            col=color(P[ok][sel],NN[sel])*sh[:,None]
        img[pix[sel]]=np.clip(col,0,1)
        zb=np.full(H*W,-np.inf); zb[pix[sel]]=z[sel]
        img=img.reshape(H,W,3); zb=zb.reshape(H,W)
        return dict(img=img,zb=zb,r=r,u=u,d=d,x0=x0,y1=y1,res=res,W=W,H=H)
    @staticmethod
    def proj(R, pts):
        pts=np.atleast_2d(pts)
        X=(pts@R['r']-R['x0'])/R['res']; Y=(R['y1']-pts@R['u'])/R['res']; Z=pts@R['d']
        return X,Y,Z
    @staticmethod
    def visible(R, pts, tol=0.6):
        X,Y,Z=Splat.proj(R,pts)
        out=[]
        for x,y,z in zip(X,Y,Z):
            ix,iy=int(x),int(y)
            if 0<=ix<R['W'] and 0<=iy<R['H']:
                win=R['zb'][max(0,iy-2):iy+3,max(0,ix-2):ix+3]
                zz=np.max(win)
                out.append(z>=zz-tol)
            else: out.append(True)
        return np.array(out)

def draw_line(ax,R,a,b,color,label=None,lw=2.0,nseg=60,vis_check=True):
    t=np.linspace(0,1,nseg)[:,None]; pts=a[None]+t*(b-a)[None]
    X,Y,Z=Splat.proj(R,pts)
    vis=Splat.visible(R,pts) if vis_check else np.ones(len(pts),bool)
    for i in range(len(pts)-1):
        ls='-' if (vis[i] and vis[i+1]) else ':'
        ax.plot(X[i:i+2],Y[i:i+2],ls,color=color,lw=lw,solid_capstyle='butt')
    if label:
        ax.text(X[-1],Y[-1],label,color=color,fontsize=8,fontweight='bold')
