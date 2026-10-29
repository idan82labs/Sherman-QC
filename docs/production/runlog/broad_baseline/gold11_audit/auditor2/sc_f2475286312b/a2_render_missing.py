"""Auditor-2 renders of formed folds that carry no gold line (sc_f2475286312b)."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, '.')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from a2_common import OUT, load_gold

fr = json.load(open(OUT / 'a2_frame.json'))
c = np.array(fr['origin']); R = np.array(fr['R_rows_u_v_n'])
q = np.load(OUT / 'q_cache.npy').astype(float)
G = {k: ((a - c) @ R.T, (b - c) @ R.T) for k, (a, b) in load_gold().items()}
ml = json.load(open(OUT / 'a2_match.json'))['reference_lines']
fig, axs = plt.subplots(2, 4, figsize=(26, 13))
axs = axs.ravel()


def sect(ax, axis, s, ij, centre, half, title, lines=()):
    a = 'uvw'.index(axis); i, j = ij
    m = (np.abs(q[:, a] - s) < 0.5) & (np.abs(q[:, i] - centre[0]) < half * 1.5) & (np.abs(q[:, j] - centre[1]) < half * 1.5)
    ax.plot(q[m, i], q[m, j], '.', ms=1.4, color='0.2')
    for k in lines:
        v = ml[k]; A = np.array(v['start_frame']); B = np.array(v['end_frame'])
        t = (s - A[a]) / (B[a] - A[a]) if abs(B[a] - A[a]) > 1e-9 else None
        if t is not None and -0.05 <= t <= 1.05:
            p = A + t * (B - A)
            ax.plot(p[i], p[j], 'D', color='#d62728', mec='k', ms=7, label='bench apex line ' + k.split('_')[1])
    ax.set_xlim(centre[0] - half, centre[0] + half); ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_title(title, fontsize=9)
    ax.set_xlabel('uvw'[i]); ax.set_ylabel('uvw'[j])
    h, l = ax.get_legend_handles_labels()
    if h: ax.legend(fontsize=7, loc='upper right')


sect(axs[0], 'v', 0.0, (0, 2), (-124, -4), 12, 'MISSING: left-edge lip (u~-126.5), section v=0; no gold line', ['BL_B4_left_lip', 'A2_F_left_lip'])
sect(axs[1], 'v', -90.0, (0, 2), (-124, -4), 12, 'MISSING: left-edge lip, section v=-90 (fold v in [-101,101])', ['BL_B4_left_lip', 'A2_F_left_lip'])
sect(axs[2], 'w', 25.0, (0, 1), (130, -108), 12, 'MISSING: -v wall ear (vertical bend), section w=25', ['BL_B7_ear_minus_v'])
sect(axs[3], 'w', 25.0, (0, 1), (130, 108), 12, 'MISSING: +v wall ear (vertical bend), section w=25', ['BL_B8_ear_plus_v'])
# top-down views of tab ends (w in 4..12): kink
for ax, (vc, nm, k) in zip(axs[4:6], ((-82, 'bottom', 'BL_B9_tab_bottom_endkink'), (92, 'top', 'BL_B10_tab_top_endkink'))):
    m = (q[:, 2] > 5) & (q[:, 2] < 12) & (np.abs(q[:, 1] - vc) < 8) & (q[:, 0] > 5) & (q[:, 0] < 40)
    ax.plot(q[m, 0], q[m, 1], '.', ms=1.2, color='0.2')
    v = ml[k]; A = np.array(v['start_frame']); B = np.array(v['end_frame'])
    ax.plot([A[0], B[0]], [A[1], B[1]], 'D', color='#d62728', mec='k', ms=7, label='bench apex line (vertical; seen end-on)')
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_xlim(5, 40); ax.set_ylim(vc - 8, vc + 8)
    ax.set_title(f'MISSING: {nm} window-tab end kink (~28-29 deg, vertical bend), top view of tab band w=5..12', fontsize=9)
    ax.set_xlabel('u'); ax.set_ylabel('v')
# tab hinges with the nearest gold
sect(axs[6], 'u', -10.0, (1, 2), (-81, 3), 14, 'bottom tab hinge (u=-10): nearest gold M03 5.7 mm / M04 8.8 mm', ['BL_B5_tab_bottom_hinge'])
sect(axs[7], 'u', -10.0, (1, 2), (86, 3), 14, 'top tab hinge (u=-10): nearest gold M05 14.8 mm / M06 15.9 mm', ['BL_B6_tab_top_hinge'])
for ax, gids, s in ((axs[6], ('M03', 'M04'), -10.0), (axs[7], ('M05', 'M06'), -10.0)):
    for g in gids:
        A, B = G[g]; t = (s - A[0]) / (B[0] - A[0]); p = A + t * (B - A)
        ax.plot(p[1], p[2], '*', color='#1a9c3a', mec='k', ms=14)
        ax.annotate(g, (p[1], p[2]), xytext=(5, 5), textcoords='offset points', color='#0b5d1e', fontweight='bold')
fig.suptitle('sc_f2475286312b auditor2: formed folds without a matching gold line (red diamond = bench_labeling apex line crossing the section)', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / 'a2_missing_folds.png', dpi=80)
print('ok')
