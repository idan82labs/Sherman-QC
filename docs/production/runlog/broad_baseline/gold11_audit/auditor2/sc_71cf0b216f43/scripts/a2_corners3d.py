"""Auditor 2: oblique 3D views of the four vertical box corners (tab / extra-fold check) with gold endpoints."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a2_frame import *
V, N, F = load_scan(); G = load_gold()
c, U = refined_frame(V, N)
P = to_frame(V, c, U); Nn = N @ U
GL = {k: (to_frame(a, c, U)[0], to_frame(b, c, U)[0]) for k, (a, b) in G.items()}
cols = {'M01':'green','M02':'olive','M03':'darkcyan','M04':'purple','M05':'red','M06':'orange','M07':'magenta','M08':'blue'}
corners = {'(-e1,-e2)': (-149, -99.2), '(+e1,-e2)': (163.9, -99.2), '(-e1,+e2)': (-149, 106.8), '(+e1,+e2)': (163.9, 106.8)}
fig = plt.figure(figsize=(24, 12))
for i, (name, (x0, y0)) in enumerate(corners.items()):
    m = (np.abs(P[:, 0]-x0) < 22) & (np.abs(P[:, 1]-y0) < 22)
    Q = P[m][::2]
    for j, (elev, az) in enumerate([(25, 45 + 90*i), (-25, 45 + 90*i + 180)]):
        ax = fig.add_subplot(2, 4, i + 1 + 4*j, projection='3d')
        ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=0.3, c=Q[:, 2], cmap='viridis')
        for k, (A, B) in GL.items():
            seg = np.linspace(0, 1, 400)[:, None]*(B-A) + A
            mm = (np.abs(seg[:, 0]-x0) < 22) & (np.abs(seg[:, 1]-y0) < 22)
            if mm.any():
                ax.plot(seg[mm, 0], seg[mm, 1], seg[mm, 2], color=cols[k], lw=2.5)
                ax.text(*seg[mm][len(seg[mm])//2], k, color=cols[k], fontsize=9)
        ax.view_init(elev=elev, azim=az); ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(x0-22, x0+22); ax.set_ylim(y0-22, y0+22); ax.set_zlim(-22, 22)
        ax.set_title(f'corner {name} elev {elev} az {az}', fontsize=9)
plt.tight_layout(); plt.savefig(OUT + '/a2_corners3d.png', dpi=55); plt.close()
print('ok')
