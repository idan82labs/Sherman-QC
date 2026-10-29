"""Auditor-2 loader for sc_71cf0b216f43 (10839000_01_full_scan.ply). Read-only on source."""
import json, os, numpy as np
SCAN = '/Users/idantal/ShermanQC/source-assets/02_CAD_QC_3D/10839000_01_full_scan.ply'
GOLD = '/Users/idantal/ShermanQC/FullProject/data/bend_corpus/annotations/bend_line_gold_catalog_v1.json'
CACHE = '/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a2_71cf'
OUT = '/Users/idantal/ShermanQC/FullProject/docs/production/runlog/broad_baseline/gold11_audit/auditor2/sc_71cf0b216f43'

def load_scan():
    vp = os.path.join(CACHE, 'V.npy'); npth = os.path.join(CACHE, 'N.npy'); fp = os.path.join(CACHE, 'F.npy')
    if os.path.exists(vp):
        return np.load(vp), np.load(npth), np.load(fp)
    with open(SCAN, 'rb') as f:
        hdr = b''
        while True:
            line = f.readline(); hdr += line
            if line.strip() == b'end_header': break
        nv = int([l for l in hdr.split(b'\n') if l.startswith(b'element vertex')][0].split()[-1])
        nf = int([l for l in hdr.split(b'\n') if l.startswith(b'element face')][0].split()[-1])
        vd = np.frombuffer(f.read(nv * 24), dtype='<f4').reshape(nv, 6)
        fd = np.frombuffer(f.read(nf * 13), dtype=np.uint8).reshape(nf, 13)
    assert (fd[:, 0] == 3).all()
    F = fd[:, 1:].copy().view('<i4').reshape(nf, 3)
    V = vd[:, :3].astype(np.float64); N = vd[:, 3:].astype(np.float64)
    np.save(vp, V); np.save(npth, N); np.save(fp, F)
    return V, N, F

def load_gold():
    d = json.load(open(GOLD))
    c = [c for c in d['cases'] if c['case_id'] == 'corpus/01_10839000'][0]
    assert c['scan_sha256'].startswith('71cf0b216f43')
    return {b['bend_id']: (np.array(b['line_start'], float), np.array(b['line_end'], float)) for b in c['bends']}
