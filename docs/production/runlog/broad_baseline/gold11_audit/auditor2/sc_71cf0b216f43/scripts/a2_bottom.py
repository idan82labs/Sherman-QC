"""Auditor 2: bottom-flange layout (points with e3<-15) coloured by height; corner zooms."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a2_frame import *
V, N, F = load_scan(); G = load_gold()
c, U = refined_frame(V, N)
P = to_frame(V, c, U); Nn = N @ U
GL = {k: (to_frame(a, c, U)[0], to_frame(b, c, U)[0]) for k, (a, b) in G.items()}
cols = {'M01':'green','M02':'olive','M03':'darkcyan','M04':'purple','M05':'red','M06':'orange','M07':'magenta','M08':'blue'}
m = (P[:, 2] < -15) & (Nn[:, 2] < -0.9)   # outer (downward-facing) skin of bottom flanges
fig, ax = plt.subplots(1, 1, figsize=(18, 12))
sc = ax.scatter(P[m, 0], P[m, 1], s=0.3, c=P[m, 2], cmap='viridis', vmin=-22.2, vmax=-19)
plt.colorbar(sc, ax=ax, label='e3 of downward-facing skin (mm)')
for k in ['M01', 'M02', 'M06', 'M08']:
    A, B = GL[k]; ax.plot([A[0], B[0]], [A[1], B[1]], color=cols[k], lw=2); ax.annotate(k, (A[0], A[1]), color=cols[k], fontsize=12)
ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title('bottom flanges: downward-facing (outer) skin, coloured by height; bottom-bend gold lines overlaid')
plt.tight_layout(); plt.savefig(OUT + '/a2_bottom_flanges.png', dpi=60); plt.close()
# corner zooms (top view of all points with e3<-12 and with e3>12), 4 corners
corners = {'(-e1,-e2)': (-150, -100), '(+e1,-e2)': (164, -100), '(-e1,+e2)': (-150, 107), '(+e1,+e2)': (164, 107)}
fig, axs = plt.subplots(2, 4, figsize=(24, 12))
for col, (name, (x0, y0)) in enumerate(corners.items()):
    for row, (lab, mm) in enumerate([('bottom (e3<-12)', P[:, 2] < -12), ('top (e3>12)', P[:, 2] > 12)]):
        ax = axs[row, col]
        s = mm & (np.abs(P[:, 0]-x0) < 25) & (np.abs(P[:, 1]-y0) < 25)
        sc = ax.scatter(P[s, 0], P[s, 1], s=0.4, c=P[s, 2], cmap='viridis')
        for k, (A, B) in GL.items():
            ax.plot([A[0], B[0]], [A[1], B[1]], color=cols[k], lw=2)
            for E in (A, B):
                if abs(E[0]-x0) < 25 and abs(E[1]-y0) < 25:
                    ax.plot(E[0], E[1], 'o', color=cols[k], mec='k'); ax.annotate(f'{k} e3={E[2]:.1f}', (E[0], E[1]), color=cols[k], fontsize=9, xytext=(4, 4), textcoords='offset points')
        ax.set_xlim(x0-25, x0+25); ax.set_ylim(y0-25, y0+25); ax.set_aspect('equal'); ax.grid(alpha=.3)
        ax.set_title(f'corner {name} {lab}', fontsize=10)
plt.tight_layout(); plt.savefig(OUT + '/a2_corners_topview.png', dpi=60); plt.close()
print('ok')
