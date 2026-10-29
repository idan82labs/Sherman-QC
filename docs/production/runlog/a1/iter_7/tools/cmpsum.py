import re, sys, collections
f = sys.argv[1]
cur=None; rows=[]
for line in open(f):
    if line.startswith("== "):
        cur=line.split()[1:4]
    m=re.match(r"(B\d+)\s+nom\s+([\d.]+)\s+off\s+([\d.]+) \(U ([\d.]+)\)\s+on\s+([\d.]+) \(U ([\d.]+)\).*shift p/c ([\d.]+)/([\d.]+)",line)
    if m and cur:
        b,nom,off,uoff,on,uon,sp,sc=m.groups()
        rows.append((cur[0],cur[1],b,float(nom),float(off),float(uoff),float(on),float(uon),float(sp),float(sc)))
fam=collections.defaultdict(list)
for r in rows: fam[r[1]].append(r)
print("family  n  MAE_off MAE_on  meanU_off meanU_on  n_changed>0.05")
to=tn=0;n=0
for k,rs in sorted(fam.items()):
    mo=sum(abs(r[4]-r[3]) for r in rs)/len(rs); mn=sum(abs(r[6]-r[3]) for r in rs)/len(rs)
    uo=sum(r[5] for r in rs)/len(rs); un=sum(r[7] for r in rs)/len(rs)
    print(f"{k:9s} {len(rs):3d}  {mo:.3f}  {mn:.3f}   {uo:.3f}  {un:.3f}  {sum(1 for r in rs if abs(r[6]-r[4])>0.05)}")
    to+=sum(abs(r[4]-r[3]) for r in rs); tn+=sum(abs(r[6]-r[3]) for r in rs); n+=len(rs)
print("ALL", n, round(to/n,3), round(tn/n,3))
for r in rows:
    if abs(r[6]-r[4])>0.2: print("  big", r[0], r[1], r[2], "nom",r[3],"off",r[4],"on",r[6],"U",r[5],"->",r[7],"shift",r[8],r[9])
