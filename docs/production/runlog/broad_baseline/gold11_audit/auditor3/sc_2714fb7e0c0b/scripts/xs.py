"""Auditor-3 independent cross-section analysis (light; numpy only)."""
import numpy as np, json, math
W='/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/'
V=np.load(W+'V.npy'); N=np.load(W+'N.npy')
G=json.load(open('/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/human_bend_lines_v1/c57b956b11cab8f4a8d3dc17.json'))
GOLD={b['bend_id']:(np.array(b['line_start']),np.array(b['line_end'])) for b in G['bends']}

def basis(a, hint=None):
    a=a/np.linalg.norm(a)
    h=np.array([0,0,1.0]) if hint is None else hint
    if abs(h@a)>0.9: h=np.array([1.0,0,0])
    e1=h-a*(h@a); e1/=np.linalg.norm(e1); e2=np.cross(a,e1)
    return a,e1,e2

class Tube:
    """points within radius rw of an infinite line (pre-filter)."""
    def __init__(self, A, B, rw=40.0, ext=60.0):
        self.A=A; self.B=B; L=np.linalg.norm(B-A); self.L=L
        self.a,self.e1,self.e2=basis(B-A)
        d=V-A; t=d@self.a
        perp=d-np.outer(t,self.a); r=np.linalg.norm(perp,axis=1)
        m=(r<rw)&(t>-ext)&(t<L+ext)
        self.P=V[m]; self.Nn=N[m]; self.t=t[m]
        self.uv=np.c_[perp[m]@self.e1, perp[m]@self.e2]
        self.nuv=np.c_[self.Nn@self.e1, self.Nn@self.e2]; self.na=self.Nn@self.a
    def slab(self, s, h=0.6):
        m=np.abs(self.t-s)<h
        return self.uv[m], self.nuv[m], self.na[m]

def two_skins(c):
    """split 1-D offsets into two skins; return (lo, hi) medians or None"""
    c=np.sort(c)
    if len(c)<10: return None
    # histogram peaks at 0.1mm
    lo,hi=c[int(0.02*len(c))],c[int(0.98*len(c))]
    if hi-lo<0.8: return (np.median(c),None)
    gaps=np.diff(c); i=np.argmax(gaps*( (c[:-1]>lo)&(c[1:]<hi) ))
    thr=(c[i]+c[i+1])/2
    a=c[c<thr]; b=c[c>=thr]
    if len(a)<5 or len(b)<5: return (np.median(c),None)
    return (np.median(a),np.median(b))

def fit_section(uv, nuv, na, g2=np.zeros(2), rmin=5.0, rmax=12.0, angtol=10.0, ref_dirs=None):
    """Fit the two legs of a bend in a 2-D section. Returns dict or None."""
    res={}
    ok=np.abs(na)<0.5
    nn=nuv[ok]; P=uv[ok]
    nrm=np.linalg.norm(nn,axis=1); good=nrm>0.5
    nn=nn[good]/nrm[good,None]; P=P[good]
    r=np.linalg.norm(P-g2,axis=1)
    phi=np.degrees(np.arctan2(nn[:,1],nn[:,0]))%180.0
    if ref_dirs is None:
        sel=(r>rmin)&(r<rmax)
        if sel.sum()<40: return None
        hist,ed=np.histogram(phi[sel],bins=90,range=(0,180))
        hs=hist+np.roll(hist,1)+np.roll(hist,-1)
        p1=np.argmax(hs)
        d=np.abs(((np.arange(90)-p1)+45)%90-45)  # circular distance in bins (2 deg each) mod 180
        hs2=hs.copy(); hs2[np.minimum(np.abs(np.arange(90)-p1),90-np.abs(np.arange(90)-p1))<8]=0
        p2=np.argmax(hs2)
        peaks=[ed[p1]+1.0, ed[p2]+1.0]
    else:
        peaks=ref_dirs
    legs=[]
    for pk in peaks:
        dphi=np.abs(((phi-pk)+90)%180-90)
        sel=(dphi<angtol)&(r>rmin)&(r<rmax)
        if sel.sum()<15: return None
        # refine angle by PCA of points? use normals mean (sign-aligned)
        nref=np.array([math.cos(math.radians(pk)),math.sin(math.radians(pk))])
        q=nn[sel]*np.sign(nn[sel]@nref)[:,None]; nmean=q.mean(0); nmean/=np.linalg.norm(nmean)
        # direction along leg away from g2
        tdir=np.array([-nmean[1],nmean[0]])
        pts=P[sel]
        if np.median((pts-g2)@tdir)<0: tdir=-tdir
        # keep only points on the ray side
        along=(pts-g2)@tdir
        pts=pts[along>rmin*0.8]
        c=pts@nmean
        sk=two_skins(c)
        legs.append(dict(n=nmean,d=tdir,skins=sk,npts=len(pts),pts=pts))
    L1,L2=legs
    # outer skin: side away from the other leg
    for A,B in ((L1,L2),(L2,L1)):
        s=np.sign(B['d']@A['n'])  # other leg on +n side => concave side is +n
        sk=A['skins']
        if sk is None: return None
        if sk[1] is None:
            A['outer']=sk[0]; A['inner']=None
        else:
            lo,hi=sk
            A['outer'],A['inner']=(lo,hi) if s>0 else (hi,lo)
        A['concave_sign']=s
    def inter(n1,c1,n2,c2):
        M=np.array([n1,n2]); 
        if abs(np.linalg.det(M))<1e-3: return None
        return np.linalg.solve(M,np.array([c1,c2]))
    Xo=inter(L1['n'],L1['outer'],L2['n'],L2['outer'])
    Xi=None
    if L1['inner'] is not None and L2['inner'] is not None:
        Xi=inter(L1['n'],L1['inner'],L2['n'],L2['inner'])
    if Xo is None: return None
    bis=-(L1['d']+L2['d']); 
    if np.linalg.norm(bis)<1e-6: return None
    b=bis/np.linalg.norm(bis)   # points to convex outside
    perpb=np.array([-b[1],b[0]])
    ang_between=math.degrees(math.acos(np.clip(L1['d']@L2['d'],-1,1)))
    bend_angle=180.0-ang_between
    # outer apex: along bisector from Xo into material
    allP=uv
    lat=(allP-Xo)@perpb; s=(allP-Xo)@(-b)
    m=(np.abs(lat)<0.35)&(s>-1.5)&(s<8)
    apex=None; s_apex=None
    if m.sum()>=3:
        sv=np.sort(s[m]); s_apex=sv[min(2,len(sv)-1)] if len(sv)<20 else np.percentile(sv,5)
        apex=Xo+(-b)*s_apex
    # thickness from skins
    th=[abs(L['skins'][1]-L['skins'][0]) if L['skins'][1] is not None else None for L in (L1,L2)]
    res=dict(Xo=Xo,Xi=Xi,b=b,perpb=perpb,apex=apex,s_apex=s_apex,bend_angle=bend_angle,legs=(L1,L2),thick=th)
    # circle fit (outer arc) using points in bend zone: those within 6mm of Xo and lateral within tangent region & outer side
    return res

def offsets(res, g2):
    Xo,b,pb,apex=res['Xo'],res['b'],res['perpb'],res['apex']
    out={}
    if apex is not None:
        d=g2-apex
        out['along_concave']=float(d@(-b)); out['across']=float(d@pb); out['d_apex']=float(np.linalg.norm(d))
    d=g2-Xo; out['d_Xo']=float(np.linalg.norm(d)); out['Xo_along_concave']=float(d@(-b))
    if res['Xi'] is not None: out['d_Xi']=float(np.linalg.norm(g2-res['Xi']))
    return out
