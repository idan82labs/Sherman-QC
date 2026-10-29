"""Shaded mesh views (orthographic, z-buffered via painter's sort) with gold lines and fold apex lines."""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from common import *

m = load_mesh(); gold = load_gold()
fr = json.load(open(OUT + '/scripts/common_frame.json'))
d, u, w, c0 = (np.array(fr[k]) for k in ('d', 'u', 'w', 'c0'))
V = m.vertices - c0
F = m.faces
FN = m.face_normals
gcol = {'M01': '#1a9c3a', 'M02': '#0060c0', 'M03': '#c03000'}


def view(ax, R, title, lims=None, light=None):
    # R rows: screen x, screen y, depth (towards viewer)
    P = V @ R.T
    tri = P[F]
    depth = tri[:, :, 2].mean(1)
    order = np.argsort(depth)
    n = FN @ R.T
    L = np.array([0.3, 0.4, 0.86]) if light is None else light
    L = L / np.linalg.norm(L)
    shade = np.clip(np.abs(n @ L), 0, 1) * 0.75 + 0.2
    cols = np.c_[shade, shade, shade * 1.02]
    cols = np.clip(cols, 0, 1)
    pc = PolyCollection(tri[order, :, :2], facecolors=cols[order], edgecolors='none', antialiased=False)
    ax.add_collection(pc)
    for gid, (a, b) in gold.items():
        pa = (a - c0) @ R.T; pb = (b - c0) @ R.T
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], '-', color=gcol[gid], lw=2.2)
        ax.text(*(pa[:2] + pb[:2]) / 2, gid, color=gcol[gid], fontsize=9, fontweight='bold')
    ax.set_aspect('equal')
    if lims:
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    else:
        ax.autoscale_view()
    ax.set_title(title, fontsize=9)


def rot(yaw, pitch):
    # base frame: x=u, y=w, z=d -> then rotate
    B = np.array([u, w, d])
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rx @ Ry @ B

views = [(rot(0, 0), 'along axis (looking -d): section-like view'),
         (rot(90, 0), 'side: screen x = -d (axis), y = w'),
         (rot(90, -90), 'from +w (top of arm): x=-d, y=u'),
         (rot(90, 90), 'from -w (bottom of arm)'),
         (rot(35, 30), 'iso 1'),
         (rot(-35, -30), 'iso 2'),
         (rot(145, 30), 'iso 3'),
         (rot(-145, -30), 'iso 4')]
fig, axs = plt.subplots(2, 4, figsize=(28, 14))
for ax, (R, t) in zip(axs.ravel(), views):
    view(ax, R, t)
fig.suptitle('sc_70b39f530f00 auditor3: shaded mesh, gold lines (M01 green, M02 blue, M03 red)')
fig.tight_layout()
fig.savefig(OUT + '/shaded_views.png', dpi=70)
print('ok')
