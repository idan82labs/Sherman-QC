"""Build benchmark_v3/inventory.json from the 2026-10-17 Codex fetch (metadata + hashes only;
no scan geometry is read)."""
import json, os, re, glob, hashlib, subprocess, collections, datetime
from pathlib import Path

S = Path('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad')
FP = Path('/Users/idantal/ShermanQC/FullProject')
DL = FP / 'data/bend_corpus/incoming_codex_fetch_2026-10-17'
V2 = FP / 'docs/production/runlog/benchmark_v2'
V3 = FP / 'docs/production/runlog/benchmark_v3'
HOME = str(Path.home())
SEALED_PARTS = {'16231000', '38168001', '38173001', '26687013', '31409000', '47215000', '42637000',
                '50476000', '31367000', '30820000'}
SEALED_PART_REVS = ['16231000_03', '38168001_02', '38173001_01', '26687013_A00', '31409000_01', '47215000_B',
                    '42637000_B', '50476000_A', '31367000_A00', '30820000_A01']
GOLD_TIERS = {'GOLD_LINES_HUMAN', 'AI_GOLD_ACCEPTED', 'GOLD_LINES_AI_LEGACY'}


def tilde(p):
    return p.replace(HOME, '~', 1) if p.startswith(HOME) else p


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


# ---------- download hashes ----------
dl = {}
for line in open(S / 'sha_dl.txt'):
    h, p = line.rstrip('\n').split('  ', 1)
    dl[p] = h
crdl = sorted(str(p) for p in DL.rglob('*.crdownload'))

manifest = json.load(open(DL / 'MANIFEST.json'))
man_sha, man_entry = {}, {}
for e in manifest:
    for p, h in (e.get('sha256') or {}).items():
        man_sha[p] = h
        man_entry[p] = e
    for p in e.get('saved_paths') or []:
        man_entry.setdefault(p, e)
# verify every manifest hash we can (cheap; the task asked for at least a sample)
ver = {'checked': 0, 'match': 0, 'mismatch': [], 'manifest_path_missing_on_disk': [], 'on_disk_not_in_manifest_hashes': []}
for p, h in man_sha.items():
    if p.endswith('.crdownload'):
        continue
    if p not in dl:
        ver['manifest_path_missing_on_disk'].append(tilde(p)); continue
    ver['checked'] += 1
    if dl[p] == h: ver['match'] += 1
    else: ver['mismatch'].append({'path': tilde(p), 'manifest': h, 'shasum': dl[p]})
for p in dl:
    if p not in man_sha and not p.endswith('MANIFEST.json'):
        ver['on_disk_not_in_manifest_hashes'].append(tilde(p))
# independent re-hash of a sample with python hashlib (the bulk pass used shasum -a 256)
sample = sorted(dl)[::25]
ver['hashlib_spotcheck'] = {'n': len(sample), 'agree': sum(sha_file(p) == dl[p] for p in sample)}

# ---------- existing corpus ----------
mi = json.load(open(V2 / 'merged_inventory.json'))
bv2 = json.load(open(V2 / 'BENCHMARK_V2.json'))
li = json.load(open(V2 / 'local_inventory.json'))
allow = {c['scan_sha256'] for c in json.load(open(FP / 'research/f1_aug30_retrospective_pseudo_unseen_train_allowlist_v1.json'))['cases']}
sealed_skus = {f['part_id'] for f in li['files'] if f['prior_use'] == 'EXCLUDED_SEALED' and f.get('part_id')}
sealed_sha = {f['sha256'] for f in li['files'] if f['prior_use'] == 'EXCLUDED_SEALED' and f['sha256'] != 'NOT_HASHED_SEALED'}
mi_by_sha = {f['sha256']: f for f in mi['files']}
v2_scan = {}
for pid, part in bv2['parts'].items():
    for s in part['scans']:
        v2_scan[s['sha256']] = (pid, s)
v2_cad = {}
for pid, part in bv2['parts'].items():
    for f in part['cad']['files']:
        v2_cad[f['sha256']] = pid
nhr = {r['scan_id']: r for r in bv2['needs_human_review']}
relabel_runs = {}
for d in ('label_lines_r2', 'label_lines_r3', 'label_prep_r3'):
    for x in (V2 / d).glob('sc_*'):
        relabel_runs.setdefault(x.name, []).append(d)

ex_sha = collections.defaultdict(list)
for line in open(S / 'sha_existing.txt'):
    h, p = line.rstrip('\n').split('  ', 1)
    ex_sha[h].append(tilde(p))

# label files -> source sha (size-matched to downloads only)
roots = {'source-assets': FP.parent / 'source-assets', 'data': FP / 'data', 'uploads': FP / 'uploads',
         'live_scans': FP / 'live_scans', 'output': FP / 'output'}
dl_sizes = {os.path.getsize(p) for p in dl}
label_by_sha = collections.defaultdict(list)
for lf in glob.glob(str(FP / 'data/bend_corpus/annotations/ai_bend_lines_v1/*.json')) + \
          glob.glob(str(FP / 'data/bend_corpus/annotations/human_bend_lines_v1/*.json')):
    d = json.load(open(lf))
    src = roots.get(d.get('source')); rel = d.get('source_relative_path')
    if not src or not rel: continue
    p = src / rel
    if p.exists() and os.path.getsize(p) in dl_sizes:
        label_by_sha[sha_file(p)].append({'label_file': tilde(lf), 'annotator': d.get('annotator'),
                                          'review_status': d.get('review_status'), 'n_bends': len(d.get('bends', []))})

# ---------- STEP results ----------
step_res = {}
for f in (V3 / 'step_v10').glob('*.json'):
    r = json.load(open(f)); step_res[f.stem] = r

# ---------- classification ----------
SCAN_EXT = {'ply', 'stl', 'obj'}
CAD_EXT = {'stp', 'step', 'sldprt', 'x_t', 'x_b', 'xdl'}
DRAW_EXT = {'pdf', 'dxf', 'dwg'}


def classify(name):
    ext = name.rsplit('.', 1)[-1].lower() if '.' in name else ''
    if ext in SCAN_EXT: return ext, 'scan'
    if ext in CAD_EXT: return ext, 'cad'
    if ext in DRAW_EXT: return ext, 'drawing'
    if ext == 'zip': return ext, 'archive'
    if name == 'MANIFEST.json': return ext, 'manifest'
    return ext, 'other'


PART_RE = re.compile(r'(?<!\d)(\d{7,8})[_ -]?([A-Za-z0-9]{0,6}(?:\.\d)?)')


def parse_part(name, entry):
    stem = name.rsplit('.', 1)[0]
    m = PART_RE.search(stem)
    if m:
        pid = m.group(1)
        rev = m.group(2) if m.group(2) and m.group(2).upper() not in ('SCAN', 'FULL') else ''
        return pid, rev, 'filename'
    subj = (entry or {}).get('subject', '') or ''
    m = re.search(r'(?<!\d)(\d{8})(?:_([A-Za-z0-9]+))?', subj)
    if m:
        return m.group(1), m.group(2) or '', 'email_subject (no SKU in filename; file co-delivered in that message)'
    return '', '', 'none'


def stage_hints(name):
    n = name.upper(); out = []
    if 'BEFORE BENDING' in n: out.append('FLAT_BEFORE_BENDING')
    if re.search(r'(?<!FULL)(?<!FULL_)SCAN_?1(?!\d)', n): out.append('PARTIAL_SCAN_1')
    if re.search(r'(?<!FULL)(?<!FULL_)SCAN_?2(?!\d)', n): out.append('PARTIAL_SCAN_2')
    if 'SECOND_SCAN' in n: out.append('SECOND_SCAN')
    if '3RD_SCAN' in n: out.append('THIRD_SCAN')
    if 'FULL_SCAN' in n or 'FULL SCAN' in n: out.append('FULL_SCAN')
    if 'PRELIM' in n or 'PREL' in n: out.append('PRELIM_REV')
    if 'RFQ' in n: out.append('RFQ')
    return out


def vx(name):
    m = re.search(r'\((V|X\d?)\)', name, re.I)
    return m.group(1).upper() if m else None


files = []
for p in sorted(dl):
    name = os.path.basename(p)
    if name == 'MANIFEST.json': continue
    if any(s in name for s in SEALED_PART_REVS) or any(s in name for s in SEALED_PARTS):
        files.append({'path': tilde(p), 'status': 'EXCLUDED_SEALED_BY_NAME_NOT_OPENED'}); continue
    h = dl[p]; ext, kind = classify(name)
    entry = man_entry.get(p)
    if entry is None:  # zip-extracted files: attach to the zip's manifest entry
        for q, e in man_entry.items():
            if e.get('extracted_directory') and p.startswith(e['extracted_directory']):
                entry = e; break
        if entry is None:
            zdir = os.path.dirname(p.split('_unzipped/')[0]) if '_unzipped/' in p else None
            for q, e in man_entry.items():
                if zdir and os.path.dirname(q) == zdir:
                    entry = e; break
    pid, rev, basis = parse_part(name, entry)
    if kind == 'cad' and ext in ('x_t', 'x_b') and not re.match(r'^\d{8}', name):
        sib = sorted({m.group(1) for q in dl if os.path.dirname(q) == os.path.dirname(p) and q != p
                      for m in [re.match(r'^(\d{8})_', os.path.basename(q))] if m})
        if len(sib) == 1:
            pid, rev, basis = sib[0], '', f'co-delivered with {sib[0]} scan in the same email (customer drawing name; filename has no Sherman SKU)'
    if kind == 'cad' and name == 'Default.stp':
        pid, rev, basis = '49125000', 'A00', 'email_subject (Default.stp delivered in the 49125000 STEP+scan message; v2 lists this sha as UNKNOWN:cad_85940820151e)'
    rec = {'path': tilde(p), 'name': name, 'sha256': h, 'size': os.path.getsize(p), 'ext': ext, 'kind': kind,
           'part_id': pid, 'rev': rev, 'part_id_basis': basis,
           'source': (entry or {}).get('source'), 'source_msg_or_link': ((entry or {}).get('link_or_msg_id') or '')[:200],
           'email_subject': (entry or {}).get('subject'), 'email_date': (entry or {}).get('date'),
           'stage_hints': stage_hints(name), 'vx': vx(name) if kind == 'scan' else None,
           'is_flat': 'BEFORE BENDING' in name.upper(), 'zip_member': '_unzipped/' in p}
    if entry and (entry.get('subject') or '').startswith('49182000'):
        rec['note'] = 'link listed under email subject 49182000_B served 42725000_B.ply; part taken from the filename'
    if entry and 'eyal_b' in json.dumps(entry):
        rec['delivery'] = 'eyal_b SharePoint (possible Eyal-50 delivery)'
    # existing-corpus matches
    m = []
    if h in mi_by_sha:
        f = mi_by_sha[h]
        m.append({'src': 'merged_inventory', 'kind': f['kind'], 'part_id': f['part_id'], 'prior_use': f['prior_use'],
                  'has_gold_lines': f['has_gold_lines'], 'eyal50': f.get('eyal50'), 'paths': f['paths'][:4]})
    if h in v2_scan:
        vp, s = v2_scan[h]
        m.append({'src': 'BENCHMARK_V2.scan', 'part_id': vp, 'scan_id': s['scan_id'], 'primary_truth': s['primary_truth'],
                  'eval_tier': s.get('eval_tier'), 'role': s['role'], 'ai_label_status': (s.get('ai_label_run') or {}).get('final_status') if isinstance(s.get('ai_label_run'), dict) else None})
    if h in v2_cad:
        m.append({'src': 'BENCHMARK_V2.cad', 'part_id': v2_cad[h]})
    if h in ex_sha:
        m.append({'src': 'size_matched_local_sha', 'paths': ex_sha[h][:6]})
    if h in label_by_sha:
        m.append({'src': 'studio_label_files', 'labels': label_by_sha[h]})
    rec['existing_matches'] = m
    rec['new_to_corpus'] = not m
    if h in sealed_sha:
        rec['status'] = 'EXCLUDED_SEALED_HOLDOUT10_HASH'
    elif '2026-08-30' in (rec['email_date'] or '') or '/20260830_' in p:
        rec['aug30_daily'] = True
        rec['aug30_train18'] = h in allow
        rec['status'] = 'OK' if (h in allow or h in mi_by_sha) else 'EXCLUDED_AUG30_UNVERIFIED_NOT_IN_TRAIN18'
    else:
        rec['status'] = 'OK'
    if kind == 'cad' and ext in ('stp', 'step'):
        r = step_res.get(h[:12])
        rec['step_v10'] = None if r is None else {
            'status': r['status'], 'v10_bend_segments': r.get('v10_bend_count'),
            'bend_lines_same_plates_merged': r.get('bend_lines_same_plates_merged'),
            'merged_groups': r.get('merged_groups'), 'lenient_bend_count': r.get('lenient_bend_count'),
            'extension_status': r.get('extension_status'), 'raw': tilde(str(V3 / 'step_v10' / f'{h[:12]}.json'))}
    files.append(rec)

ok = [f for f in files if f.get('status') == 'OK']
# within-download dedupe
by_sha = collections.defaultdict(list)
for f in ok: by_sha[f['sha256']].append(f)
for h, fs in by_sha.items():
    canon = sorted(fs, key=lambda f: (f['zip_member'], '__dup' in f['name'], len(f['path']), f['path']))[0]
    for f in fs:
        f['dup_within_download_of'] = None if f is canon else canon['path']

# ---------- parts ----------
parts = collections.defaultdict(lambda: {'files': []})
for f in ok:
    parts[f['part_id'] or f"UNKNOWN:{f['kind']}_{f['sha256'][:12]}"]['files'].append(f)
v2parts = bv2['parts']; mparts = mi['parts']
CONSUMED_USES = {'Gold11', 'Gold37', 'physical15', 'other-consumed', 'unknown'}
part_out = {}
for pid, P in sorted(parts.items()):
    fs = P['files']
    v2p = v2parts.get(pid); mp = mparts.get(pid)
    uses = set((v2p or {}).get('prior_use', [])) | set((mp or {}).get('prior_use', []))
    for f in fs:
        for m in f['existing_matches']:
            if m.get('prior_use'): uses.add(m['prior_use'])
            if m['src'] == 'BENCHMARK_V2.scan': uses.add('BENCHMARK_V2_scan_' + str(m.get('eval_tier')))
    v2split = (v2p or {}).get('split')
    if v2split in ('EXCLUDED_EYAL50_QUARANTINE', 'HELD_EYAL50_OVERLAP') or any(m.get('eyal50') for f in fs for m in f['existing_matches']):
        split, basis = 'QUARANTINE_EYAL50', f'v2 split={v2split}: Eyal-50 locked-holdout membership unread; not queued'
    elif v2split == 'CONSUMED' or uses & CONSUMED_USES or any(u.startswith('BENCHMARK_V2_scan_') for u in uses):
        split, basis = 'CONSUMED', f'v2 split={v2split}; prior_use={sorted(uses)}'
    elif v2split == 'FRESH' or v2p is None:
        split, basis = 'FRESH', ('v2 split=FRESH (no scans were in v2; never tuned/evaluated)' if v2p else
                                 'part not in BENCHMARK_V2 / merged_inventory; never used')
    else:
        split, basis = 'CONSUMED', f'v2 split={v2split} (conservative)'
    if pid.startswith('UNKNOWN:'):
        split, basis = ('FRESH', 'no SKU in name; sha unknown to corpus -> own part, never used') if not any(f['existing_matches'] for f in fs) else ('CONSUMED', 'sha known to corpus')
    steps = [f for f in fs if f.get('step_v10')]
    for c in ((v2p or {}).get('cad') or {}).get('files', []):
        if c['format'].lower() in ('stp', 'step') and c['sha256'] not in {f['sha256'] for f in fs}:
            r = step_res.get(c['sha256'][:12])
            if r:
                steps.append({'name': os.path.basename(c['path']) + ' (corpus, not in download)', 'sha256': c['sha256'],
                              'step_v10': {'status': r['status'], 'v10_bend_segments': r.get('v10_bend_count'),
                                           'bend_lines_same_plates_merged': r.get('bend_lines_same_plates_merged'),
                                           'merged_groups': r.get('merged_groups'), 'lenient_bend_count': r.get('lenient_bend_count')}})
    cad_counts = sorted({(f['step_v10']['v10_bend_segments'], f['step_v10']['bend_lines_same_plates_merged']) for f in steps if f['step_v10']['status'] == 'OK'})
    # V/X pairs per rev
    pairs = collections.defaultdict(dict)
    for f in fs:
        if f['kind'] == 'scan' and f['vx'] and not f['dup_within_download_of']:
            pairs[f['rev']].setdefault(f['vx'], []).append(f['sha256'][:12])
    for r, v in pairs.items():
        if any(k.startswith('X') for k in v) and 'V' not in v:
            untagged = sorted({f['sha256'][:12] for f in fs if f['kind'] == 'scan' and f['rev'] == r and not f['vx'] and not f['dup_within_download_of'] and f['ext'] == 'ply'})
            if len(untagged) == 1:
                v['V_presumed_untagged'] = untagged
    v2_cad_step = [c for c in ((v2p or {}).get('cad') or {}).get('files', []) if c['format'].lower() in ('stp', 'step')]
    part_out[pid] = {
        'split': split, 'split_basis': basis, 'v2_split': v2split,
        'in_gold11': (v2p or {}).get('in_gold11'), 'in_gold37': (v2p or {}).get('in_gold37'), 'in_physical15': (v2p or {}).get('in_physical15'),
        'revs': sorted({f['rev'] for f in fs}),
        'n_files': len(fs), 'n_unique_scans': len({f['sha256'] for f in fs if f['kind'] == 'scan'}),
        'kinds': dict(collections.Counter(f['kind'] for f in fs)),
        'has_step_in_download': any(f.get('path') for f in steps),
        'cad_bend_lines_low_confidence': any((f['step_v10'].get('lenient_bend_count') or 0) > (f['step_v10'].get('v10_bend_segments') or 0) for f in steps), 'has_step_in_corpus_v2': bool(v2_cad_step),
        'cad_formats': sorted({f['ext'] for f in fs if f['kind'] == 'cad'} | set(((v2p or {}).get('cad') or {}).get('formats', []))),
        'cad_bend_counts_v10': [{'step': f['name'], 'sha12': f['sha256'][:12], **{k: f['step_v10'][k] for k in ('status', 'v10_bend_segments', 'bend_lines_same_plates_merged', 'merged_groups', 'lenient_bend_count')}} for f in steps],
        'cad_bend_lines': (cad_counts[0][1] if len({c[1] for c in cad_counts}) == 1 else None) if cad_counts else None,
        'cad_bend_lines_note': None if len({c[1] for c in cad_counts}) <= 1 else f'STEPs disagree: {cad_counts}',
        'vx_pairs': {r: v for r, v in pairs.items()},
        'sealed_sku_overlap': pid in sealed_skus,
    }

# ---------- label queue ----------
scans = collections.OrderedDict()
for f in ok:
    if f['kind'] != 'scan' or f['dup_within_download_of']: continue
    scans[f['sha256']] = f
cands, skipped = [], []
for h, f in scans.items():
    pid = f['part_id'] or f"UNKNOWN:scan_{h[:12]}"
    P = part_out[pid]; sid = 'sc_' + h[:12]
    gold = []
    for m in f['existing_matches']:
        if m['src'] == 'BENCHMARK_V2.scan' and m['primary_truth'] in GOLD_TIERS: gold.append(m['primary_truth'])
        if m['src'] == 'merged_inventory' and m.get('has_gold_lines'): gold.append('merged_inventory.has_gold_lines')
        if m['src'] == 'studio_label_files':
            for l in m['labels']:
                if l['review_status'] in ('complete', 'ai_adjudicated'): gold.append(f"studio:{l['annotator']}")
    f['has_gold'] = bool(gold); f['gold_basis'] = sorted(set(gold))
    base = {'scan_id': sid, 'part_id': pid, 'rev': f['rev'], 'path': f['path'], 'split': P['split'], 'format': f['ext'],
            'size': f['size'], 'stage_hints': f['stage_hints'], 'vx': f['vx'],
            'cad_bend_lines': P['cad_bend_lines'] if P['cad_bend_lines'] is not None else -1,
            'has_step': P['has_step_in_download'] or P['has_step_in_corpus_v2']}
    if gold:
        skipped.append({**base, 'reason': 'HAS_GOLD', 'gold': sorted(set(gold))}); continue
    if P['split'] == 'QUARANTINE_EYAL50':
        skipped.append({**base, 'reason': 'EYAL50_QUARANTINE'}); continue
    roles = [m.get('role') for m in f['existing_matches'] if m['src'] == 'BENCHMARK_V2.scan']
    if 'EXCLUDED_REDACTED_OPAQUE_COPY' in roles:
        skipped.append({**base, 'reason': 'OPAQUE_COPY_CUSTODY (sha equals a redacted opaque-ID copy in v2; not queued)'}); continue
    if 'FORMAT_ALTERNATE' in roles:
        skipped.append({**base, 'reason': 'FORMAT_ALTERNATE_V2'}); continue
    if f['ext'] != 'ply':
        key = (pid, f['rev'], tuple(f['stage_hints']), f['vx'])
        twins = [g for g in scans.values() if g['ext'] == 'ply' and (g['part_id'] or '') == f['part_id'] and g['rev'] == f['rev']
                 and tuple(g['stage_hints']) == tuple(f['stage_hints']) and g['vx'] == f['vx']]
        if twins:
            skipped.append({**base, 'reason': 'FORMAT_ALTERNATE_OF_PLY', 'ply_twin': 'sc_' + twins[0]['sha256'][:12]}); continue
        base['note'] = 'non-PLY scan with no PLY twin: write-label indexes .ply only, so a PLY copy/export is needed before writing the label'
    if P['sealed_sku_overlap']:
        skipped.append({**base, 'reason': 'SKU_OVERLAPS_SEALED_SET (another file of this SKU is sealed; labelling it as CONSUMED would feed tuning; owner decides)'}); continue
    if sid in relabel_runs:
        skipped.append({**base, 'reason': 'IN_PROGRESS_' + '+'.join(relabel_runs[sid])}); continue
    if f['is_flat']:
        base['note'] = 'flat blank before bending (zero-bend control)'
    if sid in nhr:
        base['note'] = (base.get('note', '') + ' v2 NEEDS_HUMAN_REVIEW: ' + nhr[sid]['reason']).strip()
    pr = P['vx_pairs'].get(f['rev'], {})
    tag = f['vx'] or ('V_presumed' if 'sc_'[:0] + h[:12] in pr.get('V_presumed_untagged', []) else None)
    if tag:
        base['vx'] = tag
        base['pair_partners'] = ['sc_' + x for k, v in pr.items() for x in v if x != h[:12]]
    if P.get('cad_bend_lines_low_confidence') and base['cad_bend_lines'] >= 0:
        base['cad_note'] = 'V10 lenient census finds more coaxial bend candidates than V10 accepts; CAD count may be low'
    if h[:12] == '766be04dc456':
        base['note'] += '; file header says VXelements binary STL (scanner export, not CAD)'
    cands.append(base)


def tier(c):
    if c['split'] == 'FRESH': return 0
    if c['has_step']: return 1
    return 2


# prefer PLY over STL/OBJ alternates of the same capture, then V/X paired scans, then size
def prio(c):
    fmt = {'ply': 0, 'stl': 1, 'obj': 2}[c['format']]
    return (tier(c), fmt, 0 if c['vx'] else 1, c['part_id'], c['path'])


cands.sort(key=prio)
CAP = 140
queue, dropped = cands[:CAP], cands[CAP:]
inv = {
    'schema': 'sherman.benchmark_v3.inventory.v1',
    'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'download_root': tilde(str(DL)),
    'method': 'shasum -a 256 of every file (python hashlib spot-check); dedupe vs merged_inventory/BENCHMARK_V2 hashes, size-matched '
              'local files under FullProject/{data,uploads} and source-assets (sealed/holdout/opaque/evaluator_only/custody paths and '
              'sealed part numbers not hashed), studio label source files, and within the download. Part from 8-digit SKU in filename '
              '(email subject only when the filename has none). Split per BENCHMARK_V2 rules: CONSUMED if any v2 prior use; FRESH if v2 '
              'FRESH or unknown to v2. Aug-30 daily scans checked against HOLDOUT10 hashes (v2 local_inventory) and the TRAIN18 allowlist. '
              'STEP bend counts: scripts/extract_step_topology_bend_axes_v10.py run serially, one process per STEP, under the geometry lock; '
              'bend lines = V10 segments merged when collinear (<2 deg, <1 mm) and folding the same two plates (>=3 shared plane faces). '
              'No scan geometry was read.',
    'custody': {'sealed_parts_excluded_by_name': SEALED_PART_REVS, 'manifest_skipped_sealed': sum(e['status'] == 'SKIPPED_SEALED' for e in manifest),
                'files_excluded_by_name': [f['path'] for f in files if f.get('status') == 'EXCLUDED_SEALED_BY_NAME_NOT_OPENED'],
                'files_excluded_holdout10_hash_count': sum(1 for f in files if f.get('status') == 'EXCLUDED_SEALED_HOLDOUT10_HASH'),
                'holdout10_note': 'aug30 daily-scan downloads whose sha256 equals a HOLDOUT10 file (v2 local_inventory EXCLUDED_SEALED hashes); excluded, not opened, and deliberately not listed by path or part here',
                'files_excluded_aug30_unverified': [f['path'] for f in files if f.get('status') == 'EXCLUDED_AUG30_UNVERIFIED_NOT_IN_TRAIN18'],
                'crdownload_incomplete_ignored': [tilde(p) for p in crdl]},
    'hash_verification': ver,
    'summary': {},
    'parts': part_out,
    'files': [f for f in files if f.get('status') != 'EXCLUDED_SEALED_HOLDOUT10_HASH'],
    'label_queue': {'cap': CAP, 'ordering': 'FRESH, then CONSUMED parts with a STEP (download or corpus), then the rest; within a tier PLY before STL/OBJ, V/X-paired first',
                    'n_candidates': len(cands), 'queue': queue, 'dropped_over_cap': dropped, 'not_queued': skipped},
}
inv['summary'] = {
    'files_total_hashed': len(files), 'files_ok': len(ok),
    'by_kind': dict(collections.Counter(f['kind'] for f in ok)),
    'unique_sha_ok': len({f['sha256'] for f in ok}),
    'dup_within_download': sum(1 for f in ok if f['dup_within_download_of']),
    'files_new_to_corpus': sum(1 for f in ok if f['new_to_corpus']),
    'unique_scans': len(scans), 'unique_scans_new_to_corpus': sum(1 for f in scans.values() if f['new_to_corpus']),
    'parts': len(part_out), 'parts_by_split': dict(collections.Counter(p['split'] for p in part_out.values())),
    'steps_run': sum(1 for f in ok if f.get('step_v10')), 'parts_with_step_in_download': sum(p['has_step_in_download'] for p in part_out.values()),
    'parts_with_vx_pair': sum(1 for p in part_out.values() if any('V' in v and any(k.startswith('X') for k in v) for v in p['vx_pairs'].values())),
    'queue': len(queue), 'queue_by_split': dict(collections.Counter(q['split'] for q in queue)),
    'dropped_over_cap': len(dropped), 'not_queued': dict(collections.Counter(s['reason'] for s in skipped)),
}
V3.mkdir(parents=True, exist_ok=True)
(V3 / 'inventory.json').write_text(json.dumps(inv, indent=1, ensure_ascii=False))
print(json.dumps(inv['summary'], indent=1)); print(json.dumps({k: v for k, v in ver.items() if k != 'on_disk_not_in_manifest_hashes'}, indent=1)[:1500])
print('not in manifest hashes:', ver['on_disk_not_in_manifest_hashes'][:20])
