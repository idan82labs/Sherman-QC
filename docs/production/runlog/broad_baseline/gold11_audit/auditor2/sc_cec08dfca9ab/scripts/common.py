import numpy as np, json
S=np.load("/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2/scan.npz")
V=S["V"]; F=S["F"]; N=S["N"]
G=json.load(open("/Users/idantal/ShermanQC/FullProject/output/bend_label_03_10839000/bend_instance_labels.json"))
gold={b["bend_id"]:(np.array(b["line_start"]),np.array(b["line_end"])) for b in G["manual_bends"]}
def unit(v): v=np.asarray(v,float); return v/np.linalg.norm(v)
a=unit(gold["HB1"][1]-gold["HB1"][0])
b0=unit(gold["HB3"][1]-gold["HB3"][0])
b=unit(b0-(b0@a)*a)
c=np.cross(a,b)
R=np.stack([a,b,c])  # rows: X,Y,Z axes
O=V.mean(0)
def tof(P): return (np.asarray(P)-O)@R.T
VF=tof(V)
