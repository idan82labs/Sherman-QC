#!/usr/bin/env python3
"""Native-resolution V32 source-free visual evidence renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


BG = "#070b12"
INK = "#e8edf5"
MUTED = "#8d9bad"
A_COLOR = "#20d7b2"
B_COLOR = "#ffb454"
PASS = "#53e081"
FAIL = "#ff5d6c"
CURRENT = "#4cb8ff"


def _candidate(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["candidate_records"][0]


def _frame(candidate: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(candidate["center_xyz"], dtype=np.float64)
    axis = np.asarray(candidate["unit_axis_xyz"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    n0 = np.asarray(candidate["parent_A"]["unit_normal_xyz"], dtype=np.float64)
    radial = n0 - np.dot(n0, axis) * axis
    radial /= np.linalg.norm(radial)
    tangent = np.cross(axis, radial)
    tangent /= np.linalg.norm(tangent)
    return center, axis, radial, tangent


def _style(ax: Any, title: str) -> None:
    ax.set_facecolor(BG)
    ax.set_title(title, color=INK, fontsize=18, pad=12)
    ax.tick_params(colors=MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#263447")
    ax.grid(color="#1b2634", linewidth=0.4, alpha=0.7)


def _scatter(ax: Any, x: np.ndarray, y: np.ndarray, parents: np.ndarray, size: float = 0.8, alpha: float = 0.75) -> None:
    colors = np.where(parents == 1, A_COLOR, np.where(parents == 2, B_COLOR, MUTED))
    ax.scatter(x, y, c=colors, s=size, alpha=alpha, linewidths=0, rasterized=True)


def render_raw_multiview(payload: dict[str, Any], output_path: Path) -> Path:
    points = np.asarray(payload["source"]["positions"], dtype=np.float64)
    parents = np.asarray(payload["source"]["parent_fragment_ids"], dtype=np.int64)
    candidate = _candidate(payload)
    center, axis, radial, tangent = _frame(candidate)
    rel = points - center
    z, r, t = rel @ axis, rel @ radial, rel @ tangent
    fig, axes = plt.subplots(2, 2, figsize=(20.48, 20.48), dpi=200, facecolor=BG)
    views = ((z, r, "axis × parent-A normal"), (z, t, "axis × transverse"), (r, t, "cross-section"), (points[:, 0], points[:, 2], "world X × Z"))
    for ax, (u, v, title) in zip(axes.ravel(), views):
        _style(ax, title)
        _scatter(ax, u, v, parents)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"V32 raw source geometry — {payload.get('fixture_id','fixture')}", color=INK, fontsize=28, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=BG)
    plt.close(fig)
    return output_path


def render_axial_unwrapped_input(payload: dict[str, Any], output_path: Path) -> Path:
    points = np.asarray(payload["source"]["positions"], dtype=np.float64)
    parents = np.asarray(payload["source"]["parent_fragment_ids"], dtype=np.int64)
    candidate = _candidate(payload)
    center, axis, radial, tangent = _frame(candidate)
    scale = float(candidate["local_scale_s"])
    rel = (points - center) / scale
    z, r, t = rel @ axis, rel @ radial, rel @ tangent
    n0 = np.asarray(candidate["parent_A"]["unit_normal_xyz"], dtype=np.float64)
    n1 = np.asarray(candidate["parent_B"]["unit_normal_xyz"], dtype=np.float64)
    p0 = points @ n0 - float(candidate["parent_A"]["offset"])
    p1 = points @ n1 - float(candidate["parent_B"]["offset"])
    electrode0 = (parents == int(candidate["parent_A"]["parent_id"])) & (np.abs(p0) <= 0.75 * scale)
    electrode1 = (parents == int(candidate["parent_B"]["parent_id"])) & (np.abs(p1) <= 0.75 * scale)
    fig, axes = plt.subplots(3, 1, figsize=(20.48, 20.48), dpi=200, facecolor=BG, sharex=True)
    _style(axes[0], "Unwrapped source support and frozen tail memberships")
    axes[0].scatter(z, r, c=MUTED, s=0.5, alpha=0.35, linewidths=0, rasterized=True)
    axes[0].scatter(z[electrode0], r[electrode0], c=A_COLOR, s=1.2, alpha=0.9, linewidths=0, rasterized=True, label="F0 input predicate")
    axes[0].scatter(z[electrode1], r[electrode1], c=B_COLOR, s=1.2, alpha=0.9, linewidths=0, rasterized=True, label="F1 input predicate")
    axes[0].legend(facecolor="#101824", edgecolor="#2c3d52", labelcolor=INK, loc="upper right")
    _style(axes[1], "Opposite-skin and transverse support")
    _scatter(axes[1], z, t, parents, size=0.7)
    _style(axes[2], "Source-derived axial support density")
    bins = np.linspace(-13.2, 13.2, 133)
    axes[2].hist(z[(z >= -13.2) & (z <= 13.2)], bins=bins, color="#7387a4", alpha=0.85)
    axes[2].set_ylabel("vertices", color=MUTED)
    axes[2].set_xlabel("normalized axial coordinate z/s", color=MUTED)
    for ax in axes:
        ax.axvspan(-12, 12, color="#36577d", alpha=0.08)
        ax.axvline(-13.2, color=FAIL, linestyle="--", linewidth=1)
        ax.axvline(13.2, color=FAIL, linestyle="--", linewidth=1)
        ax.axvline(-12, color=CURRENT, linestyle=":", linewidth=1.2)
        ax.axvline(12, color=CURRENT, linestyle=":", linewidth=1.2)
        ax.set_xlim(-16.5, 16.5)
    fig.suptitle(f"V32 axial input audit — {payload.get('fixture_id','fixture')} — no detector decision", color=INK, fontsize=27, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=BG)
    plt.close(fig)
    return output_path


def render_stress_page(payload: dict[str, Any], page: int, output_path: Path) -> Path:
    points = np.asarray(payload["source"]["positions"], dtype=np.float64)
    candidates = payload["candidate_records"][page * 16:(page + 1) * 16]
    fig, axes = plt.subplots(4, 4, figsize=(20.48, 20.48), dpi=200, facecolor=BG)
    for ax, candidate in zip(axes.ravel(), candidates):
        _style(ax, f"ordinal {candidate['frozen_v16_order']}")
        center, axis, radial, _tangent = _frame(candidate)
        rel = points - center
        z, r = rel @ axis, rel @ radial
        mask = (np.abs(z) <= 14) & (np.abs(r) <= 24)
        ax.scatter(z[mask], r[mask], c=MUTED, s=0.4, alpha=0.55, linewidths=0, rasterized=True)
        ax.axvline(-12, color=CURRENT, linestyle=":", linewidth=0.8)
        ax.axvline(12, color=CURRENT, linestyle=":", linewidth=0.8)
        ax.set_xlim(-14, 14)
    fig.suptitle(f"V32 stress preflight — page {page + 1}/8 — candidates {page*16:03d}–{page*16+15:03d}", color=INK, fontsize=24, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=BG)
    plt.close(fig)
    return output_path


def _visual(result: dict[str, Any]) -> dict[str, Any]:
    values = result.get("_ephemeral", {}).get("candidate_visuals", [])
    return values[0] if values else {"indices": [], "phi": [], "current_edges": [], "events": [], "intervals": []}


def render_harmonic_current_multiview(payload: dict[str, Any], result: dict[str, Any], output_path: Path) -> Path:
    points = np.asarray(payload["source"]["positions"], dtype=np.float64)
    candidate = _candidate(payload)
    center, axis, radial, tangent = _frame(candidate)
    visual = _visual(result)
    indices = np.asarray(visual.get("indices", []), dtype=np.int64)
    fig, axes = plt.subplots(2, 2, figsize=(20.48, 20.48), dpi=200, facecolor=BG)
    for ax, horiz, vert, title in ((axes[0,0],axis,radial,"axis × parent normal"),(axes[0,1],axis,tangent,"axis × transverse"),(axes[1,0],radial,tangent,"cross-section"),(axes[1,1],axis,radial,"current detour audit")):
        _style(ax, title)
        rel = points - center
        ax.scatter(rel[:,0]*0 + rel@horiz, rel@vert, c=MUTED, s=0.35, alpha=0.18, linewidths=0, rasterized=True)
        if len(indices) and len(visual.get("phi", [])):
            local = points[indices] - center
            phi = np.asarray(visual["phi"], dtype=float)
            sc = ax.scatter(local@horiz, local@vert, c=phi, cmap="turbo", s=1.1, vmin=0, vmax=1, linewidths=0, rasterized=True)
            segments=[]; widths=[]
            for edge in visual.get("current_edges", []):
                segments.append([[float(local[edge['i']]@horiz),float(local[edge['i']]@vert)],[float(local[edge['j']]@horiz),float(local[edge['j']]@vert)]])
                widths.append(min(2.5,0.2+40*float(edge['current'])))
            if segments:
                ax.add_collection(LineCollection(segments, colors=CURRENT, linewidths=widths, alpha=0.35))
    fig.colorbar(sc if len(indices) else plt.cm.ScalarMappable(cmap="turbo"), ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label="quantized potential")
    fig.suptitle(f"V32 harmonic current — {payload.get('fixture_id','fixture')}", color=INK, fontsize=27, y=0.985)
    fig.tight_layout(rect=(0,0,0.97,0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=BG)
    plt.close(fig)
    return output_path


def render_axial_measure_audit(payload: dict[str, Any], result: dict[str, Any], output_path: Path) -> Path:
    visual = _visual(result)
    events = visual.get("events", [])
    decision = result.get("decisions", [{}])[0]
    fig, axes = plt.subplots(2, 1, figsize=(20.48, 20.48), dpi=200, facecolor=BG, gridspec_kw={"height_ratios":[2,1]})
    _style(axes[0], "Event cells and local terminal reasons")
    reason_names=sorted(set(str(e.get("reason")) for e in events)); reason_color={r:plt.cm.tab20(i%20) for i,r in enumerate(reason_names)}
    for event in events:
        left,right=np.asarray(event["cell_q21"],dtype=float)/(1<<21)
        axes[0].barh(0.0,right-left,left=left,height=0.72,color=reason_color[str(event["reason"])],edgecolor="none")
    for i,r in enumerate(reason_names): axes[0].text(-11.8,1.0+i*0.35,r,color=reason_color[r],fontsize=11)
    axes[0].set_xlim(-12,12); axes[0].set_ylim(-1,1+0.4*max(1,len(reason_names)))
    _style(axes[1], "Nested accepted axial measures")
    stages=list(decision.get("stage_measures_q21",{})); vals=[decision["stage_measures_q21"][s]/(1<<21) for s in stages]
    axes[1].bar(np.arange(len(stages)),vals,color=[PASS if v>=19.5 else FAIL for v in vals]); axes[1].axhline(19.5,color=INK,linestyle="--")
    axes[1].set_xticks(np.arange(len(stages)),stages,rotation=22,ha="right",color=MUTED); axes[1].set_ylim(0,24.5)
    axes[1].text(0.01,0.92,f"final={decision.get('accepted_length_q21',0)/(1<<21):.3f}s  longest={decision.get('longest_run_q21',0)/(1<<21):.3f}s  gap={decision.get('maximum_gap_q21',0)/(1<<21):.3f}s  reason={decision.get('reason')}",transform=axes[1].transAxes,color=INK,fontsize=14)
    fig.suptitle(f"V32 axial measure audit — {payload.get('fixture_id','fixture')}",color=INK,fontsize=27,y=0.99)
    fig.tight_layout(rect=(0,0,1,0.97)); output_path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output_path,facecolor=BG); plt.close(fig); return output_path


def render_final_overlay(payload: dict[str, Any], result: dict[str, Any], output_path: Path) -> Path:
    points=np.asarray(payload["source"]["positions"],dtype=float); parents=np.asarray(payload["source"]["parent_fragment_ids"],dtype=int); candidate=_candidate(payload); center,axis,radial,tangent=_frame(candidate); rel=points-center
    fig,axes=plt.subplots(1,2,figsize=(20.48,20.48),dpi=200,facecolor=BG)
    for ax,h,v,title in ((axes[0],axis,radial,"axial overlay"),(axes[1],radial,tangent,"cross-section overlay")):
        _style(ax,title); _scatter(ax,rel@h,rel@v,parents,0.7,0.5)
        start=np.asarray(candidate["render_line_start"])-center; end=np.asarray(candidate["render_line_end"])-center
        ax.plot([start@h,end@h],[start@v,end@v],color=PASS if result.get("pre_nms_rescue_count") else FAIL,linewidth=5,alpha=0.9)
        ax.set_aspect("equal",adjustable="datalim")
    fig.suptitle(f"V32 unchanged V16 line / NMS outcome — {payload.get('fixture_id','fixture')} — pre={result.get('pre_nms_rescue_count')} final={result.get('rescue_count')}",color=INK,fontsize=24,y=0.985)
    fig.tight_layout(rect=(0,0,1,0.97)); output_path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output_path,facecolor=BG); plt.close(fig); return output_path


def render_raw_fixture(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    return render_raw_multiview(payload, output_dir/"raw_multiview.png"), render_axial_unwrapped_input(payload, output_dir/"axial_unwrapped_input.png")


def render_post_fixture(payload: dict[str, Any], result: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    return (render_harmonic_current_multiview(payload,result,output_dir/"harmonic_current_multiview.png"),render_axial_measure_audit(payload,result,output_dir/"axial_measure_audit.png"),render_final_overlay(payload,result,output_dir/"final_overlay.png"))
