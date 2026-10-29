"""User-requested single consumed-case report; preserves all earlier artifacts."""
from pathlib import Path
import sys, json, time, os, signal, subprocess, resource, hashlib, base64, html

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reports/2026-09-23_37361005_ply_vs_cad'
sys.path.insert(0, str(ROOT))

def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def worker():
    resource.setrlimit(resource.RLIMIT_CPU, (480, 480))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import open3d as o3d
    from scipy.spatial import cKDTree
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from scripts import run_gold11_v11_single_case_candidate_v1 as candidate
    from backend.cad_guided_cad_surface_v2 import sample_native_cad_surface_v2

    print('Fresh single-case extraction and unchanged matching', flush=True)
    result = candidate.run(OUT / 'extraction')
    score = result['score']
    assert score['true_positive'] == score['gold_count'] == 4
    assert score['false_positive'] == score['false_negative'] == 0
    retained = json.loads(candidate.RETAINED_V11_RESULT.read_text())
    transform = candidate._scan_to_cad_from_retained_v11(retained)
    scan = candidate.prepare_mesh_scan_v1(scan_bytes=candidate.SCAN_PATH.read_bytes(), max_engine_rows=12000)
    assert scan['status'] == 'AVAILABLE'
    raw = scan['raw_points']
    points = raw @ transform[:3, :3].T + transform[:3, 3]
    print(f'CAD triangulation and unsigned surface distances for {len(points)} vertices', flush=True)
    reader = STEPControl_Reader()
    assert reader.ReadFile(str(candidate.CAD_PATH)) == IFSelect_RetDone
    assert reader.TransferRoots() > 0
    surface = sample_native_cad_surface_v2(reader.OneShape(), allow_ocp=True, linear_deflection_mm=.10)
    assert surface['status'] == 'AVAILABLE', surface['reasons']
    triangles = surface['triangles']
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(triangles.reshape(-1, 3).astype(np.float32)),
        o3d.core.Tensor(np.arange(len(triangles)*3).reshape(-1,3).astype(np.uint32)))
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(mesh)
    distances = np.concatenate([scene.compute_distance(o3d.core.Tensor(p.astype(np.float32)), nthreads=1).numpy()
                                for p in np.array_split(points, max(1, (len(points)+49999)//50000))])
    assert len(distances) == len(raw) and np.isfinite(distances).all()
    def stats(values):
        return {'count': len(values), 'mean_mm': float(np.mean(values)), 'rms_mm': float(np.sqrt(np.mean(values**2))),
                'median_mm': float(np.median(values)), 'p90_mm': float(np.quantile(values,.9)),
                'p95_mm': float(np.quantile(values,.95)), 'p99_mm': float(np.quantile(values,.99)), 'max_mm': float(np.max(values))}
    global_stats = stats(distances)
    np.savez_compressed(OUT / 'surface_distances.npz', scan_to_cad=transform, scan_points_cad_mm=points, unsigned_distance_mm=distances)
    tree = cKDTree(raw)
    matches = {m['prediction_id']:m for m in score['matches']}
    rows, profiles = [], []
    for i, prediction in enumerate(result['recovery']['predictions']):
        line = np.array([prediction['line_start'],prediction['line_end']])
        delta = line[1]-line[0]
        t = np.linspace(0,1,201)
        profile = tree.query(line[0]+t[:,None]*delta,workers=1)[0]
        profiles.append(profile)
        u = np.clip((raw-line[0]) @ delta / (delta@delta),0,1)
        mask = np.linalg.norm(raw-(line[0]+u[:,None]*delta),axis=1)<=3.0
        rows.append({'label':f'B{i+1}', **prediction, 'gold_match':matches[prediction['bend_id']],
                     'local_surface_distance_within_3mm_of_marker':stats(distances[mask]) if mask.any() else None,
                     'measured_scan_angle_deg':None,'measured_scan_radius_mm':None})
    cmap = 'magma'
    vmax = max(1.,float(np.quantile(distances,.99)))
    sample = np.linspace(0,len(points)-1,min(len(points),40000),dtype=int)
    cloud = points[sample]
    fig = plt.figure(figsize=(15,7),dpi=160)
    for panel,(elev,azim,title) in enumerate([(24,-58,'Oblique view'),(90,-90,'Plan view')]):
        if panel == 1:
            ax = fig.add_subplot(1,2,2)
            sc = ax.scatter(cloud[:,0],cloud[:,2],c=distances[sample],s=1,cmap=cmap,vmin=0,vmax=vmax,linewidths=0)
            for row in rows:
                line = np.array([row['line_start'],row['line_end']]) @ transform[:3,:3].T + transform[:3,3]
                ax.plot(line[:,0],line[:,2],c='#00a7b5',lw=1.5)
                mid=line.mean(axis=0)
                ax.annotate(row['label'],(mid[0],mid[2]),xytext=(7,7),textcoords='offset points',fontweight='bold',color='#005563',bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=2))
            ax.set_aspect('equal'); ax.set_xlabel('CAD X (mm)'); ax.set_ylabel('CAD Z (mm)'); ax.set_title(title); ax.margins(.12)
            continue
        ax = fig.add_subplot(1,2,panel+1,projection='3d')
        sc = ax.scatter(cloud[:,0],cloud[:,2],cloud[:,1],c=distances[sample],s=1,cmap=cmap,vmin=0,vmax=vmax,linewidths=0)
        for row in rows:
            line = np.array([row['line_start'],row['line_end']]) @ transform[:3,:3].T + transform[:3,3]
            ax.plot(line[:,0],line[:,2],line[:,1],c='#00a7b5',lw=2)
            mid = line.mean(axis=0)
            ax.text(mid[0],mid[2],mid[1]+4,row['label'],fontsize=10,color='#005563',fontweight='bold')
        center = (points.max(axis=0)+points.min(axis=0))/2
        radius = np.ptp(points,axis=0).max()/2
        ax.set_xlim(center[0]-radius,center[0]+radius); ax.set_ylim(center[2]-radius,center[2]+radius); ax.set_zlim(center[1]-radius,center[1]+radius)
        ax.set_box_aspect((1,1,1)); ax.view_init(elev,azim)
        ax.set_xlabel('CAD X (mm)'); ax.set_ylabel('CAD Z (mm)'); ax.set_zlabel('CAD Y (mm)'); ax.set_title(title)
    fig.colorbar(sc,ax=fig.axes,shrink=.65,pad=.04,label=f'Unsigned scan → CAD mesh distance (mm); clipped at {vmax:.2f}',extend='max')
    fig.suptitle('37361005 • PLY versus CAD surface heatmap',fontsize=17)
    fig.text(.5,.02,'Retained registration • all vertices measured • sampled display • approximate CAD mesh • no pass/fail tolerance',ha='center',fontsize=9)
    fig.savefig(OUT/'surface_heatmap.png',bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4),dpi=160)
    im=ax.imshow(profiles,aspect='auto',origin='upper',extent=[0,100,4.5,.5],cmap=cmap,vmin=0,vmax=3)
    ax.set_yticks([1,2,3,4],['B1','B2','B3','B4']); ax.set_xlabel('Position along finite bend marker (%)')
    ax.set_title('Bend support profile • distance from marker samples to nearest raw scan point')
    fig.colorbar(im,ax=ax,label='Distance (mm); support gate ≤ 3 mm')
    fig.tight_layout(); fig.savefig(OUT/'bend_support_heatmap.png'); plt.close(fig)
    metadata = {'part':'37361005','created':'2026-09-23','score':score,'input_bindings':result['input_bindings'],
      'scan_to_cad':transform.tolist(),'registration':'Retained V11 context; no fresh alignment or fit',
      'surface_statistics':global_stats,'cad_triangle_count':len(triangles),'mesh_linear_deflection_mm':.1,
      'distance_semantics':'Unsigned scan vertex to closest triangulated CAD surface; approximate, uncalibrated, vertex-weighted, scan-to-CAD only',
      'bend_rows':rows,'scan_angle_radius_status':'UNAVAILABLE','conformance_status':'UNKNOWN',
      'production_use_allowed':False,'floor_release_allowed':False}
    write_json('analysis.json',metadata)
    table=['| Bend / reference | Supported span mm | Support | Axis→scan median / P90 mm | Label offset mm / angle ° | Local surface median / P95 mm |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        b,m,s=r['bend_data'],r['gold_match'],r['local_surface_distance_within_3mm_of_marker']
        table.append(f"| {r['label']} / {m['gold_id']} | {b['line_span_mm']:.1f} | {100*b['support_fraction']:.0f}% | {b['median_scan_distance_mm']:.3f} / {b['p90_scan_distance_mm']:.3f} | {m['transverse_offset_mm']:.3f} / {m['axis_angle_deg']:.3f} | {s['median_mm']:.3f} / {s['p95_mm']:.3f} |")
    report = f'''# PLY versus CAD report — 37361005

23 September 2026 · CG4BDD1371B6 · Research inspection

## Extraction result

Fresh execution recovered **4/4 labelled bends: 100% recall, 100% precision, F1 1.00**, with zero unmatched labels or predictions. This meets the requested >80% extraction criterion on this known consumed case. The method is CAD-guided scan support, not independent PLY-only detection. It proposes STEP bend markers through the retained V11 pose and accepts finite markers supported by raw scan points. The earlier frozen V11 attempt was not rerun or modified.

The unchanged one-to-one matcher allows axis angle ≤5°, transverse offset ≤4 mm, and overlap ≥60% in both directions. This detection score does not mean 100% dimensional accuracy or manufacturing conformance.

## Bend-by-bend analysis

{chr(10).join(table)}

All four CAD references have nominal cylindrical radii **0.8 / 2.3 mm** and nominal sweeps **90° / 90°**. These are paired CAD surface properties. Measured scan bend radius, bend angle, radius error, angle error and tolerance pass/fail are **unavailable** because the retained measurement path did not establish valid flange-plane measurements. The axis-to-label angle above measures marker orientation agreement, not bend angle error.

Support is evaluated at 201 uniformly spaced samples per bend marker within a 3 mm nearest-point gate. All 201 samples on each marker pass, and each marker has one continuous sampled support run. This does not prove continuous physical coverage between samples. Supported lengths are CAD-directed spans retained by the support test, not independently fitted manufacturing lengths. Local surface statistics use raw scan vertices within 3 mm of each finite marker; these neighborhoods can overlap.

B4 has the weakest marker support (median 1.314 mm; P90 1.851 mm), though all sampled positions pass the support gate. B2 has the largest label offset (3.779 mm, close to the 4 mm matching limit) and extends beyond its manually labelled span (150.0 versus 124.9 mm). B4's label offset is 3.395 mm. B1 and B3 have smaller label offsets, 0.848 and 0.498 mm. These observations identify review priorities; they do not establish defects.

![Bend support heatmap](bend_support_heatmap.png)

## Surface-distance heatmap

All **{len(points):,} raw PLY vertices** were transformed using the retained scan-to-CAD rigid pose and compared to **{len(triangles):,} CAD triangles**. No outliers were removed or alignment optimized for this report. Display uses {len(sample):,} deterministic samples; statistics include every vertex.

| Surface distance statistic | mm |
|---|---:|
| Mean | {global_stats['mean_mm']:.3f} |
| RMS | {global_stats['rms_mm']:.3f} |
| Median | {global_stats['median_mm']:.3f} |
| P90 | {global_stats['p90_mm']:.3f} |
| P95 | {global_stats['p95_mm']:.3f} |
| P99 | {global_stats['p99_mm']:.3f} |
| Maximum | {global_stats['max_mm']:.3f} |

![Scan to CAD surface heatmap](surface_heatmap.png)

Colors show unsigned closest-surface distance, not signed material excess/removal, bend-angle error or a calibrated defect measure. The CAD mesher uses 0.10 mm linear and 0.20 rad angular deflection; the approximation error is not certified. Color range ends at {vmax:.3f} mm; higher distances share the top color, while full values remain in the data. Statistics are weighted by scan vertex density and include registration error and acquisition artifacts. This scan-to-CAD comparison does not measure unobserved CAD areas or prove whole-part coverage.

## Disposition and traceability

**Bend localization criterion met; dimensional conformance remains unknown.** No manufacturing tolerance or calibrated scan uncertainty was provided. This report preserves the research-only status and does not issue production release. A metrology conclusion requires valid measured bend angles/radii, qualified registration and applicable tolerances.

Inputs: `{candidate.SCAN_PATH.name}` and `{candidate.CAD_PATH.name}`. Both were checked against their established SHA-256 bindings before extraction. Gold labels were consumed only after inference by the unchanged matcher. `analysis.json` includes input bindings, the transform, all matches and local statistics. `surface_distances.npz` contains every transformed scan vertex and its unsigned distance. `extraction/RESULT.json` retains the fresh extraction record. `CHECKSUMS.json` binds the report artifacts and generator.
'''
    (OUT/'REPORT.md').write_text(report)
    # A self-contained, portable HTML report, with tables and both plots embedded.
    blocks=[]; lines=report.splitlines(); idx=0
    while idx<len(lines):
        line=lines[idx]
        if line.startswith('|'):
            tablelines=[]
            while idx<len(lines) and lines[idx].startswith('|'):
                tablelines.append(lines[idx]); idx+=1
            blocks.append('<table>')
            for j,tr in enumerate(tablelines):
                if j==1: continue
                tag='th' if j==0 else 'td'
                blocks.append('<tr>'+''.join(f'<{tag}>{html.escape(c.strip())}</{tag}>' for c in tr.strip('|').split('|'))+'</tr>')
            blocks.append('</table>'); continue
        if line.startswith('!['):
            filename=line.split('](')[1][:-1]
            blocks.append('<img alt="'+html.escape(filename)+'" src="data:image/png;base64,'+base64.b64encode((OUT/filename).read_bytes()).decode()+'">')
        elif line.startswith('# '): blocks.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '): blocks.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line: blocks.append('<p>'+html.escape(line).replace('**','')+'</p>')
        idx+=1
    (OUT/'REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>37361005 PLY vs CAD report</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#182631}h1,h2{line-height:1.2}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #d3dce2}th{background:#edf3f5}img{width:100%;height:auto}p{overflow-wrap:anywhere}@media print{body{margin:0}img,table{break-inside:avoid}}</style><body>'+''.join(blocks)+'</body></html>')
    write_json('CHECKSUMS.json',{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('CHECKSUMS.json','run.log','RUN_RECEIPT.json')})
    print(json.dumps({'score':score,'surface':global_stats,'output':str(OUT)}),flush=True)

def supervise():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'REPORT.html').exists(): raise RuntimeError('Report exists; refusing overwrite')
    start=time.monotonic(); peak=0; reason=None
    with (OUT/'run.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MPLCONFIGDIR':'/tmp/sherman-report-mpl'})
        while proc.poll() is None:
            listing=subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True)
            processes=[tuple(map(int,line.split())) for line in listing.splitlines()]
            owned={proc.pid}
            for _ in processes:
                added={pid for pid,ppid,rss in processes if ppid in owned}-owned
                if not added: break
                owned.update(added)
            rss=sum(rss for pid,ppid,rss in processes if pid in owned); peak=max(peak,rss)
            if rss>16*1024**2 or time.monotonic()-start>600:
                reason='MEMORY_OR_WALL_BUDGET'; os.killpg(proc.pid,signal.SIGKILL); break
            time.sleep(.5)
        code=proc.wait()
    receipt={'exit_code':code,'elapsed_seconds':time.monotonic()-start,'peak_polled_tree_rss_kib':peak,'poll_seconds':.5,'wall_limit_seconds':600,'worker_cpu_limit_seconds':480,'rss_limit_gib':16,'stop_reason':reason,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json('RUN_RECEIPT.json',receipt); print(json.dumps(receipt))
    if code: print((OUT/'run.log').read_text()[-6000:])
    return code

if __name__=='__main__':
    if '--worker' in sys.argv: worker()
    else: raise SystemExit(supervise())
