"""Unmarked anonymous native-face review panels from released neutral evidence."""
from pathlib import Path
import argparse
import hashlib
import json


def validate_post_release(attempt, post, post_hash):
    raw = post.read_bytes()
    review = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != post_hash or review.get('decision') != 'RELEASE_NEUTRAL_EVIDENCE' or review.get('independent_non_author') is not True:
        raise ValueError('Independent evidence release required')
    pins = review['evidence_sha256']
    if set(pins) != {f'case-{i:02d}' for i in range(1, 21)}:
        raise ValueError('Incomplete neutral release')
    for name, digest in pins.items():
        p = attempt / name / 'EVIDENCE.json'
        if p.is_symlink() or not p.is_file() or p.stat().st_size > 40_000_000 or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Evidence pin mismatch')
    return pins


def render(attempt, out, post, post_hash):
    pins = validate_post_release(attempt, post, post_hash)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    out.mkdir(parents=False, exist_ok=False)
    manifest = []
    views = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 1), (-1, -1, -1)]
    for i in range(1, 21):
        name = f'case-{i:02d}'
        p = attempt / name / 'EVIDENCE.json'
        data = p.read_bytes()
        if len(data) > 40_000_000 or hashlib.sha256(data).hexdigest() != pins[name]:
            raise ValueError('Evidence pin mismatch')
        evidence = json.loads(data)
        faces = evidence['faces']
        all_points = [np.asarray(f['display_mesh']['vertices'], dtype=float) for f in faces if f.get('display_mesh')]
        if not all_points:
            raise ValueError('No display geometry')
        points = np.concatenate(all_points)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        for ax, direction in zip(axs.flat, views):
            z = np.asarray(direction, dtype=float); z /= np.linalg.norm(z)
            seed = np.eye(3)[int(np.argmin(np.abs(z)))]
            x = np.cross(seed, z); x /= np.linalg.norm(x)
            y = np.cross(z, x)
            basis = np.stack([x, y, z], axis=1)
            polys, depths, colors, labels = [], [], [], []
            for face in faces:
                mesh = face.get('display_mesh')
                if not mesh:
                    continue
                v = (np.asarray(mesh['vertices']) - center) @ basis
                t = np.asarray(mesh['triangles_one_based'], dtype=int) - 1
                if len(t) > 200_000 or t.min() < 0 or t.max() >= len(v):
                    raise ValueError('Mesh budget or indices')
                tri = v[t]
                normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                length = np.linalg.norm(normal, axis=1)
                light = .55 + .4 * np.divide(np.abs(normal[:, 2]), length, out=np.zeros(len(length)), where=length>0)
                surface = face['surface_class']
                base = np.array([.86, .65, .28] if surface == 'Cylinder' else [.43, .69, .75] if surface != 'Plane' else [.77, .8, .83])
                colors.extend(base[None, :] * light[:, None]); polys.extend(tri[:, :, :2]); depths.extend(tri[:, :, 2].mean(axis=1))
                if surface != 'Plane' and 'area_centroid' in face:
                    c = (np.asarray(face['area_centroid']) - center) @ basis
                    labels.append((c, str(face['id'])))
            order = np.argsort(depths)
            ax.add_collection(PolyCollection(np.asarray(polys)[order], facecolors=np.asarray(colors)[order], edgecolors='none'))
            projected = (points-center) @ basis
            lo, hi = projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)
            pad = max(hi-lo)*.08 + 1e-6
            ax.set_xlim(lo[0]-pad, hi[0]+pad); ax.set_ylim(lo[1]-pad, hi[1]+pad)
            for c, label in labels:
                ax.text(c[0], c[1], label, fontsize=5, color='#7b1414')
            ax.set_aspect('equal'); ax.axis('off'); ax.set_title(f'View {direction}', fontsize=9)
        fig.suptitle(f'{name} — native surface IDs only; no predicted bends', fontsize=14)
        fig.text(.02, .015, 'Orange: cylindrical surfaces (not necessarily bends). Blue: other nonplanar surfaces. Meshes aid review, not exact proof.', fontsize=9)
        fig.tight_layout(rect=(0, .03, 1, .96))
        target = out / f'{name}.png'; fig.savefig(target, dpi=150); plt.close(fig)
        manifest.append({'case': name, 'evidence_sha256': pins[name], 'image': target.name, 'image_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    with (out/'MANIFEST.json').open('x') as stream:
        json.dump({'panels': manifest, 'post_sha256': post_hash, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'prediction_access': False}, stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--post', type=Path, required=True)
    parser.add_argument('--post-sha256', required=True)
    args = parser.parse_args()
    render(args.attempt, args.out, args.post, args.post_sha256)
