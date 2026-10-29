"""Render one deterministic synthetic geometry PNG to an explicit output path.

No inputs, detectors, services or default artifact directories. The Matplotlib
font/config cache is temporary; the only retained output is --output. Run from
the Git root with PYTHONDONTWRITEBYTECODE=1 and the existing application Python.
"""

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


def render(output):
    # Import graphics only after main() has isolated Matplotlib's writable cache.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research.geometry.analytic_bend_fixture import (
        BendSpec, sample_surfaces, surface_generator,
    )

    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "axes.titlesize": 14})
    face_color, centre_color, ruling_color, virtual_color = (
        "#93b8c8", "#be6529", "#007f83", "#9a437d",
    )
    hypothetical_colors = ("#7763a5", "#c28132")
    axial = np.linspace(0, 1, 7)

    def draw_faces(ax, spec, *, color, alpha, patches=("start", "bend", "end"),
                   bend_fractions=None, flange_fractions=None, dashed=False, zorder=1):
        arc = np.linspace(0, 1, 33) if bend_fractions is None else bend_fractions
        flange = np.linspace(0, 1, 5) if flange_fractions is None else flange_fractions
        samples = sample_surfaces(spec, axial_fractions=axial, bend_fractions=arc,
                                  flange_fractions=flange, patches=patches)
        for surface in ("inner", "outer"):
            for name in patches:
                mask = (samples.surfaces == surface) & (samples.patches == name)
                grid = samples.points[mask].reshape(len(axial), -1, 3)
                ax.plot_surface(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2],
                                color=color, alpha=alpha, linewidth=0, shade=False,
                                antialiased=True, zorder=zorder)
                # Sparse boundaries communicate finite support without a dense fake scan.
                for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1]):
                    ax.plot(*edge.T, color=color, alpha=min(1, alpha * 3), linewidth=0.8,
                            linestyle="--" if dashed else "-", zorder=zorder + 1)
        return samples

    def finite_line(ax, points, color, linestyle, marker):
        ax.plot(*points.T, color=color, linestyle=linestyle, linewidth=2.7,
                marker=marker, markersize=5, markerfacecolor="white", zorder=15)

    figure = plt.figure(figsize=(15, 8), facecolor="white")
    axes = [figure.add_subplot(1, 2, index, projection="3d", computed_zorder=False)
            for index in (1, 2)]
    figure.subplots_adjust(left=0.025, right=0.985, bottom=0.25, top=0.84, wspace=0.06)
    figure.suptitle("Analytic bend geometry and visibility", fontsize=18, y=0.965)
    figure.text(0.5, 0.914, "Construction references are not detected markers or measured bend evidence.",
                ha="center", fontsize=11, color="#444444")

    # Panel A: the centre axis, inner-face ruling and virtual inner-plane intersection.
    spec = BendSpec(3, 0.6, math.pi / 2, (-3, 3), (8, 8))
    draw_faces(axes[0], spec, color=face_color, alpha=0.40)
    centre = spec.axis_segment()
    ruling = surface_generator(spec, surface="inner")
    virtual = centre + spec.inner_radius * np.array([1, 0, 1])
    finite_line(axes[0], centre, centre_color, "--", "o")
    finite_line(axes[0], ruling, ruling_color, "-", "s")
    finite_line(axes[0], virtual, virtual_color, ":", "x")
    axes[0].set_title("A  |  Three distinct finite lines", pad=18)
    axes[0].text2D(0.5, 0.96, "Inner R = 3.0 mm; thickness = 0.6 mm; turn = 90 deg",
                   transform=axes[0].transAxes, ha="center", fontsize=10)
    axes[0].legend(handles=[
        Patch(facecolor=face_color, alpha=0.6, label="Ideal inner / outer faces and tangent flanges"),
        Line2D([], [], color=centre_color, linestyle="--", marker="o", markerfacecolor="white",
               label="Centre-axis segment (construction reference)"),
        Line2D([], [], color=ruling_color, marker="s", markerfacecolor="white",
               label="Ruling on inner face (finite, not a detected marker)"),
        Line2D([], [], color=virtual_color, linestyle=":", marker="x",
               label="Virtual intersection of infinite inner flange planes"),
    ], loc="upper left", bbox_to_anchor=(0.03, -0.12), frameon=False, fontsize=10)

    # Panel B: whole hidden models are stipulated hypotheses; only dark patches observed.
    observations = []
    for radius, color in zip((2.0, 4.0), hypothetical_colors):
        hidden = BendSpec(radius, 1, math.pi / 2, (-3, 3), (10 - radius, 10 - radius),
                          origin=(-radius, 0, -radius), component_id="same-visible-patches")
        visible_start = (8 - radius) / (10 - radius)
        draw_faces(axes[1], hidden, color=color, alpha=0.10, dashed=True,
                   flange_fractions=np.linspace(0, visible_start, 5))
        finite_line(axes[1], hidden.axis_segment(), color, "--", "o")
        observations.append(sample_surfaces(hidden, axial_fractions=axial,
                                            patches=("start", "end"),
                                            flange_fractions=np.linspace(visible_start, 1, 5)))
        if radius == 2:
            observed_spec, observed_start = hidden, visible_start
    for field in ("points", "normals"):
        if not np.allclose(getattr(observations[0], field), getattr(observations[1], field),
                           atol=1e-12, rtol=0):
            raise RuntimeError(f"counterexample no longer has identical observed {field}")
    draw_faces(axes[1], observed_spec, color="#252b32", alpha=0.88,
               patches=("start", "end"),
               flange_fractions=np.linspace(observed_start, 1, 5), zorder=20)
    axes[1].set_title("B  |  Same observations, different hidden bends", pad=18)
    axes[1].text2D(0.5, 0.96, "Both face sides; thickness = 1.0 mm; shared finite axial span",
                   transform=axes[1].transAxes, ha="center", fontsize=10)
    axes[1].legend(handles=[
        Patch(facecolor="#252b32", label="Identical observed patches: x or z from -10 to -8 mm"),
        Line2D([], [], color=hypothetical_colors[0], linestyle="--", marker="o",
               markerfacecolor="white", label="Hypothesis only: hidden R = 2 mm, axis at (-2, y, -2)"),
        Line2D([], [], color=hypothetical_colors[1], linestyle="--", marker="o",
               markerfacecolor="white", label="Hypothesis only: hidden R = 4 mm, axis at (-4, y, -4)"),
    ], loc="upper left", bbox_to_anchor=(0.03, -0.12), frameon=False, fontsize=10)

    for ax in axes:
        # Same limits and orthographic viewpoint in both panels, with equal physical
        # scale along x/y/z: box aspect is proportional to the actual axis ranges.
        ax.set_xlim(-11, 5)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-11, 5)
        ax.set_box_aspect((16, 8, 16))
        ax.set_proj_type("ortho")
        ax.view_init(elev=20, azim=-57)
        ax.set_xticks([-10, -5, 0, 5])
        ax.set_yticks([-3, 0, 3])
        ax.set_zticks([-10, -5, 0, 5])
        ax.set_xlabel("x [mm]", labelpad=8)
        ax.set_ylabel("y [mm]", labelpad=8)
        ax.set_zlabel("z [mm]", labelpad=8)
        ax.tick_params(labelsize=9, pad=1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
    figure.text(0.5, 0.035,
                "Equal spatial scaling and shared limits. Ghosted geometry is hypothetical, not observed. "
                "CAD may fix nominal geometry; an unseen actual radius remains unverified.",
                ha="center", fontsize=10, color="#444444")
    figure.savefig(output, dpi=160, facecolor=figure.get_facecolor(),
                   metadata={"Title": "Analytic finite-bend diagnostic", "Software": "Sherman QC synthetic fixture"})
    plt.close(figure)
    print(f"Saved synthetic diagnostic: {output}")
    print("Verified: observed points agree within 1e-12 mm; unit normals within 1e-12 (dimensionless).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Explicit path for a new PNG; parent must exist")
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    if output.suffix.lower() != ".png":
        parser.error("--output must have a .png suffix")
    if not output.parent.is_dir():
        parser.error("--output parent must already exist; no default directory is created")
    if output.exists():
        parser.error("--output already exists; choose a new path")
    with tempfile.TemporaryDirectory(prefix="sherman-analytic-mpl-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        render(output)


if __name__ == "__main__":
    main()
