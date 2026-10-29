"""Authored charts/text and mocked processes only; NEVER native/CAD/ps work.

Kernel geometry acceptance is independently held elsewhere. These tests isolate
the new protocol's oracles, transport invocation, custody and bounded failure
paths. Every source-profile lookup and process monitor is replaced in tests.
"""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_vendor_unfold_finite_width_native_v1 as runner


@pytest.fixture
def prepared(monkeypatch,tmp_path):
    profiles=({"authored_source":"a"*64},{runner.LEGACY_PROFILE:"b"*64,runner.FINITE_PROFILE:"c"*64})
    monkeypatch.setattr(runner,"_profile_sources",lambda:copy.deepcopy(profiles))
    monkeypatch.setattr(runner.protocol,"_review",lambda *args:{"review_sha256":"d"*64,
        "source_archive_sha256":"e"*64,"independent_review_identity_declared_not_cryptographically_verified":True})
    monkeypatch.setattr(runner.legacy_worker,"ADMISSION_LOCK_PATH",str(tmp_path/"authored.lock"))
    monkeypatch.setattr(runner.legacy_worker,"_cache",{})
    monkeypatch.setattr(runner.finite_worker,"_cache",{})
    directory,record=tmp_path/"generated",tmp_path/"generation.jsonl"
    frozen=runner.freeze_generation(directory,record)
    def supervise(folder,request,lock,started,evidence):
        assert runner.protocol.strict_json((folder/"request.json").read_bytes())==request
        rows=[]
        for name in runner.CASE_IDS:
            raw=("AUTHORED FAKE TEXT, NOT CAD "+name).encode()
            case=runner.case_definition(name)
            (Path(request["freeze"]["directory"])/case["step_filename"]).write_bytes(raw)
            rows.append({**case,"source_sha256":runner._hash(raw)})
        runner.save_new(Path(request["freeze"]["directory"])/"generated-cases.json",rows)
        (folder/"stdout.log").write_bytes(b"authored stdout")
        (folder/"stderr.log").write_bytes(b"")
        evidence.update(pid=12345,pgid=12345,process_started=True,process_monitor_preflight=True,
            cleanup_quiescent=True,returncode=0,peak_process_group_rss_kib=100)
    monkeypatch.setattr(runner,"_supervise_generation",supervise)
    return directory,record,frozen


def generated(prepared):
    directory,record,frozen=prepared
    ledger=runner.generate(frozen,directory,record,"mock-pre","mock-archive")
    assert ledger["status"]=="COMPLETE_SYNTHETIC_GENERATION"
    return directory,record,ledger


def chart_receipt(check,case,bundle):
    finite=check["profile"]==runner.FINITE_PROFILE
    diag={"mechanism_version":4 if finite else 3,"transport_version":3 if finite else 2,
        "source_bundle_sha256":bundle,"candidate_binding":copy.deepcopy(case["candidates"]),
        "candidate_input_sha256":runner._hash(runner._json(case["candidates"])),"cache_hit":False,
        "elapsed_seconds":.1,"peak_process_group_rss_kib":100,
        "transitive_runtime_closure_verified":False,"hostile_upload_sandbox_verified":False,
        "limits":dict(runner.finite_worker.LIMITS if finite else runner.legacy_worker.LIMITS)}
    if finite:diag.update(cache_profile=runner.FINITE_PROFILE,v4_engine_source_sha256=runner.finite_worker.ENGINE_SHA256,
        transitive_runtime_closure_verified=False,hostile_upload_sandbox_verified=False)
    result={"status":check["expected_status"],"reason":check.get("expected_reason") if not finite else
        None if check["expected_status"]=="AVAILABLE" else "SPLIT_OR_AMBIGUOUS_TANGENT_SEAM",
        "model":"SELECTED_CAD_SKIN_DEVELOPMENT","units":"assumed_mm","units_verified":False,
        "manufacturing_ready":False,"source_sha256":case["source_sha256"],"vertices":[],"triangles":[],
        "bend_lines":[],"flange_surfaces":[],"diagnostics":diag}
    if result["status"]=="AVAILABLE":
        # Independent analytic rectangles: complete root + two tabs, quarter
        # bend strip and child. This is NOT a native or common-admission receipt.
        for x0,x1,y0,y1 in [(-20,0,-8,48),(0,8,-8,-1),(0,8,41,48),(0,runner.math.pi,0,40),
                             (runner.math.pi,runner.math.pi+15,0,40)]:
            offset=len(result["vertices"])//3
            result["vertices"].extend([x0,y0,0,x1,y0,0,x1,y1,0,x0,y1,0])
            result["triangles"].extend([offset,offset+1,offset+2,offset,offset+2,offset+3])
        result["bend_lines"]=[{"feature_id":"synthetic:B1","line_start":[runner.math.pi/2,0,0],
            "line_end":[runner.math.pi/2,40,0],"bend_direction":"unknown","sweep_deg":90.}]
    return result


def mock_evaluation(monkeypatch,frozen):
    calls=[]
    monkeypatch.setattr(runner,"validate_legacy",lambda *args:None)
    monkeypatch.setattr(runner,"validate_finite",lambda *args:None)
    def call(started,check,case,path,row):
        assert runner._hash(Path(path).read_bytes())==case["source_sha256"]
        calls.append(check["check_id"]);row["actual_call_elapsed_seconds"]=.1
        return chart_receipt(check,case,frozen["runtime_source_bundles"][check["profile"]])
    monkeypatch.setattr(runner,"_call_admitted",call)
    return calls


def test_import_is_lazy_and_native_construction_is_independent():
    source=Path(runner.__file__).read_text();tree=ast.parse(source)
    top=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
    names=[alias.name for node in top if isinstance(node,ast.Import) for alias in node.names]
    names += [node.module for node in top if isinstance(node,ast.ImportFrom)]
    assert not any(n.split('.')[0] in {'OCP','numpy','FreeCAD','Part'} for n in names)
    native=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_generate_native')
    assert any(isinstance(n,ast.ImportFrom) and n.module.startswith('OCP.') for n in ast.walk(native))
    assert 'base=fresh_base()' in source and '.copy(' not in ast.get_source_segment(source,native)
    assert runner.legacy_worker._slot is runner.finite_worker._slot


def test_fixed_footprints_oracles_and_whitelist_are_not_adaptive():
    positive,negative=[runner.case_definition(n) for n in runner.CASE_IDS]
    assert len(positive['construction']['parent_footprint_xz'])==8
    assert len(negative['construction']['parent_footprint_xz'])==12
    assert positive['oracle']['flat_area_mm2']==1232+600+40*runner.math.pi
    assert positive['oracle']['sorted_flat_bbox_mm']==[35+runner.math.pi,56]
    assert negative['oracle']['finite_guard_execution_claimed'] is False
    assert positive['candidates']==negative['candidates']
    assert [c['check_id'] for c in runner.CHECKS]==['legacy_outside_span_control','v4_outside_span_positive','v4_in_span_refusal']
    assert not any(any(x in reason for x in ('BUDGET','HASH','CAD_CANDIDATE','SOURCE','DEPENDENCY','WORKER')) for reason in runner.NATIVE_REFUSALS)
    legacy=(runner.ROOT/'backend/vendor_unfold_preview.py').read_text()
    helper=(runner.ROOT/'backend/vendor_unfold_finite_width.py').read_text()
    literals={n.value for n in ast.walk(ast.parse(legacy)) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    for reason in runner.NATIVE_REFUSALS:
        if reason.startswith('FINITE_WIDTH_'): assert reason.split(':')[1] in helper
        else: assert reason in literals
    changed=runner.case_definition(runner.CASE_IDS[0]);changed['construction']['parent_footprint_xz'][0][0]=100
    assert runner.case_definition(runner.CASE_IDS[0])['construction']['parent_footprint_xz'][0][0]==-20


def test_independent_polygon_area_relief_and_slab_geometry():
    def area(poly):
        return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(poly,poly[1:]+poly[:1])))/2
    def clip(poly, bound, lower):
        result=[]
        for a,b in zip(poly[-1:]+poly[:-1],poly):
            inside_a=a[1]>=bound if lower else a[1]<=bound
            inside_b=b[1]>=bound if lower else b[1]<=bound
            if inside_a != inside_b:
                fraction=(bound-a[1])/(b[1]-a[1])
                result.append([a[0]+fraction*(b[0]-a[0]),bound])
            if inside_b:result.append(b)
        return result
    positive,negative=[runner.case_definition(n)['construction']['parent_footprint_xz'] for n in runner.CASE_IDS]
    assert area(positive)==1232 and area(negative)==1392
    assert max(x for x,z in positive)==8  # Old global parent guard refuses.
    for poly,maximum in ((positive,0),(negative,8)):
        clipped=clip(clip(poly,-1e-5,True),40+1e-5,False)
        assert max(x for x,z in clipped)==maximum
    # Independent development chart: root material with positive x is wholly
    # outside the axial strip, and child starts AFTER the complete R2 arc.
    assert all(z<=-1 or z>=41 for x,z in positive if x>0)
    assert 0 < runner.math.pi < runner.math.pi+15
    assert area(positive)+40*runner.math.pi+15*40==pytest.approx(1957.6637061435918)


def test_source_union_profiles_are_separate_and_all_local_helpers_bound(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    for name in (*runner.OWN_SOURCES,*runner.protocol.OWN_SOURCES):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'AUTHORED SOURCE '+name.encode())
    legacy={'authored-shared':'a'*64,'authored-legacy':'b'*64}
    finite={'authored-shared':'a'*64,'authored-finite':'c'*64}
    monkeypatch.setattr(runner.legacy_worker,'_source_pins',lambda:legacy)
    monkeypatch.setattr(runner.finite_worker,'_source_pins',lambda:finite)
    union,bundles=runner._profile_sources()
    assert set(union)==set(legacy)|set(finite)|{str((tmp_path/n).resolve()) for n in (*runner.OWN_SOURCES,*runner.protocol.OWN_SOURCES)}
    assert bundles=={runner.LEGACY_PROFILE:runner._hash(runner._json(legacy)),runner.FINITE_PROFILE:runner._hash(runner._json(finite))}
    finite['authored-shared']='d'*64
    with pytest.raises(ValueError,match='SOURCE_CHANGED_BETWEEN_PROFILES'):runner._profile_sources()


def test_source_union_has_a_fixed_cap_without_runtime_hashing(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    for name in (*runner.OWN_SOURCES,*runner.protocol.OWN_SOURCES):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'AUTHORED')
    monkeypatch.setattr(runner.legacy_worker,'_source_pins',lambda:{str(i):'a'*64 for i in range(2001)})
    monkeypatch.setattr(runner.finite_worker,'_source_pins',lambda:{})
    with pytest.raises(ValueError,match='DECLARED_SOURCE_PIN_BUDGET_EXCEEDED'):runner._profile_sources()


def test_generation_is_exclusive_complete_and_evidence_survives_tempcleanup(prepared,tmp_path):
    directory,record,ledger=generated(prepared)
    assert ledger['selected_case_ids']==list(runner.CASE_IDS)
    assert len(ledger['cases'])==2 and ledger['child']['cleanup_quiescent'] is True
    for log in ledger['child']['retained_logs']:
        raw=Path(log['path']).read_bytes();assert runner._hash(raw)==log['retained_prefix_sha256']
        assert len(raw)==log['retained_prefix_bytes']
    frozen=runner.freeze_evaluation(directory,record,tmp_path/'evaluation.jsonl')
    assert frozen['generation_provenance']['record_sha256']==runner._hash(record.read_bytes())
    assert frozen['checks']==list(runner.CHECKS)
    with pytest.raises(FileExistsError):runner.generate(prepared[2],directory,record,'mock','mock')


@pytest.mark.parametrize('field,value',[('process_started',False),('cleanup_quiescent',False),
    ('returncode',True),('pid',True),('pgid',True),('pgid',12345.),('peak_process_group_rss_kib',0)])
def test_generation_provenance_refuses_unsupervised_or_nonquiescent_claims(prepared,field,value):
    directory,record,ledger=generated(prepared)
    events=[json.loads(line) for line in record.read_bytes().splitlines()]
    events[-1]['child'][field]=value
    record.write_bytes(b''.join(runner._json(event)+b'\n' for event in events))
    with pytest.raises(ValueError,match='COMPLETE_GENERATION_PROVENANCE_REQUIRED'):
        runner._generation_provenance(record,directory,ledger['cases'])


@pytest.mark.parametrize('review',[None,{}, {'review_sha256':'d'*64,'source_archive_sha256':'e'*64,
    'independent_review_identity_declared_not_cryptographically_verified':False},
    {'review_sha256':'D'*64,'source_archive_sha256':'e'*64,
    'independent_review_identity_declared_not_cryptographically_verified':True},
    {'review_sha256':'d'*64,'source_archive_sha256':True,
    'independent_review_identity_declared_not_cryptographically_verified':True},
    {'review_sha256':'d'*64,'source_archive_sha256':'e'*64,
    'independent_review_identity_declared_not_cryptographically_verified':1},
    {'review_sha256':'d'*64,'source_archive_sha256':'e'*64,
    'independent_review_identity_declared_not_cryptographically_verified':True,'extra':False}])
def test_generation_provenance_requires_exact_typed_retained_review_receipt(prepared,review):
    directory,record,ledger=generated(prepared)
    events=[json.loads(line) for line in record.read_bytes().splitlines()]
    events[0]['review']=review
    record.write_bytes(b''.join(runner._json(event)+b'\n' for event in events))
    with pytest.raises(ValueError,match='COMPLETE_GENERATION_PROVENANCE_REQUIRED'):
        runner._generation_provenance(record,directory,ledger['cases'])


@pytest.mark.parametrize('field',['directory','runner_path','limits','cases','runtime_source_bundles'])
def test_modified_generation_freeze_never_admits(prepared,field):
    directory,record,frozen=prepared;frozen[field]=None
    with pytest.raises(ValueError,match='GENERATION_FREEZE_MISMATCH'):
        runner.generate(frozen,directory,record,'mock','mock')
    assert not record.exists() and not directory.exists()


def test_failure_pid_and_both_logs_are_retained_before_cleanup(prepared,monkeypatch):
    directory,record,frozen=prepared
    def fail(folder,request,lock,started,evidence):
        (folder/'stdout.log').write_bytes(b'authored partial stdout')
        (folder/'stderr.log').write_bytes(b'authored signal evidence')
        evidence.update(pid=12345,pgid=12345,process_started=True,returncode=-11,cleanup_quiescent=True)
        raise ValueError('GENERATION_CHILD_FAILED')
    monkeypatch.setattr(runner,'_supervise_generation',fail)
    ledger=runner.generate(frozen,directory,record,'mock','mock')
    assert ledger['status']=='FAILED' and ledger['child']['returncode']==-11
    assert [Path(v['path']).read_bytes() for v in ledger['child']['retained_logs']]==[b'authored partial stdout',b'authored signal evidence']
    assert len(record.read_bytes().splitlines())==2


def test_output_validation_failure_also_retains_logs(prepared,monkeypatch):
    directory,record,frozen=prepared
    monkeypatch.setattr(runner,'load_cases',lambda *_:(_ for _ in ()).throw(ValueError('AUTHORED_INVALID_METADATA')))
    ledger=runner.generate(frozen,directory,record,'mock','mock')
    assert ledger['status']=='FAILED' and ledger['child']['retained_logs']


@pytest.mark.parametrize('state',[{'state':'ADMITTING','pgid':None},{'state':'QUARANTINED','pgid':12345}])
def test_persistent_quarantine_prevents_generation(prepared,state):
    directory,record,frozen=prepared
    Path(runner.legacy_worker.ADMISSION_LOCK_PATH).write_text(json.dumps(state))
    ledger=runner.generate(frozen,directory,record,'mock','mock')
    assert ledger['status']=='FAILED' and not directory.exists() and ledger['child']=={}


def test_shared_slot_busy_never_creates_child(prepared):
    directory,record,frozen=prepared;runner.finite_worker._slot.acquire()
    try:ledger=runner.generate(frozen,directory,record,'mock','mock')
    finally:runner.finite_worker._slot.release()
    assert ledger['reason']=='GENERATION_SHARED_SLOT_BUSY' and not directory.exists()


@pytest.mark.parametrize('mode',['denied','empty','bad','expired','slow'])
def test_preflight_monitor_and_deadline_prevent_any_spawn(monkeypatch,tmp_path,mode):
    monkeypatch.setattr(runner,'_profile_sources',lambda:({},{}))
    spawned=[];monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**k:spawned.append(k))
    clock={'now':61. if mode=='expired' else 0.}
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock['now'])
    def rows(_):
        if mode=='denied':raise PermissionError('AUTHORED DENIED')
        if mode=='slow':clock['now']=61.;return [(os.getpid(),1)]
        return [] if mode=='empty' else [(os.getpid(),True)] if mode=='bad' else [(os.getpid(),1)]
    monkeypatch.setattr(runner.legacy_worker,'_group_rows',rows)
    with (tmp_path/'mock.lock').open('w+b') as lock:
        with pytest.raises(ValueError):runner._supervise_generation(tmp_path,{'nonce':'f'*48,'freeze':{'source_pins':{},'runtime_source_bundles':{}}},lock,0.,{})
    assert not spawned


@pytest.mark.parametrize('mode,reason',[('rss','RSS_BUDGET'),('monitor','RSS_MONITOR'),('log','LOG_BUDGET'),
    ('descendants','LIVE_DESCENDANTS'),('crash','CHILD_FAILED'),('cleanup','AUTHORED_CLEANUP')])
def test_whole_group_caps_env_and_cleanup_are_mandatory(monkeypatch,tmp_path,mode,reason):
    monkeypatch.setattr(runner,'_profile_sources',lambda:({},{}))
    monkeypatch.setattr(runner,'_preflight_monitor',lambda:None)
    monkeypatch.setattr(runner,'_check_generation_output',lambda *_:None)
    polls=iter([None,0])
    process=SimpleNamespace(pid=12345,returncode=1 if mode=='crash' else 0,
        poll=lambda:None if mode in {'rss','monitor','log'} else next(polls))
    captured,stopped={},[]
    def spawn(*args,**kwargs):
        captured.update(kwargs)
        if mode=='log':kwargs['stderr'].write(b'x'*1_000_001)
        return process
    monkeypatch.setattr(runner.subprocess,'Popen',spawn)
    calls=[]
    def rows(_):
        calls.append(True)
        if mode=='monitor':return []
        if mode=='rss':return [(12345,4*1024*1024+1)]
        if mode=='descendants' and len(calls)>1:return [(12346,100)]
        return [(12345,100)]
    monkeypatch.setattr(runner.legacy_worker,'_group_rows',rows)
    def stop(p):
        stopped.append(p.pid)
        if mode=='cleanup':raise ValueError('AUTHORED_CLEANUP')
    monkeypatch.setattr(runner.legacy_worker,'_stop_group',stop)
    monkeypatch.setattr(runner.time,'sleep',lambda *_:None)
    monkeypatch.setenv('DYLD_LIBRARY_PATH','must-not-inherit')
    evidence={}
    with (tmp_path/'mock.lock').open('w+b') as lock:
        with pytest.raises(ValueError,match=reason):runner._supervise_generation(tmp_path,
            {'nonce':'f'*48,'freeze':{'source_pins':{},'runtime_source_bundles':{},'directory':str(tmp_path)}},lock,runner.time.monotonic(),evidence)
        lock.seek(0);state=json.loads(lock.read())
    assert stopped==[12345] and evidence['pid']==12345
    assert state['state']==('QUARANTINED' if mode=='cleanup' else 'QUIESCENT')
    assert captured['start_new_session'] is True and captured['env']['OMP_NUM_THREADS']=='1'
    assert 'DYLD_LIBRARY_PATH' not in captured['env'] and captured['env']['HOME']==str(tmp_path)


def test_cleanup_and_slot_release_are_inside_generation_clock(prepared,monkeypatch):
    directory,record,frozen=prepared;clock={'now':0.}
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock['now'])
    class Slot:
        def acquire(self,**kw):return True
        def release(self):clock['now']=61.
    monkeypatch.setattr(runner.legacy_worker,'_slot',Slot())
    ledger=runner.generate(frozen,directory,record,'mock','mock')
    assert ledger['status']=='FAILED' and ledger['elapsed_seconds']==61.
    assert ledger['child']['retained_logs']


@pytest.mark.parametrize('kind',['subset','oracle','hash','extra','budget'])
def test_generated_cases_are_complete_exact_and_bounded(prepared,monkeypatch,kind):
    directory,_,ledger=generated(prepared);cases=copy.deepcopy(ledger['cases'])
    if kind=='subset':cases.pop()
    elif kind=='oracle':cases[0]['oracle']['flat_area_mm2']+=1
    elif kind=='hash':cases[0]['source_sha256']='0'*64
    elif kind=='extra':(directory/'not-selected.step').write_bytes(b'AUTHORED')
    else:monkeypatch.setattr(runner,'MAX_STEP_BYTES',1)
    (directory/'generated-cases.json').write_bytes(runner._json(cases))
    with pytest.raises(ValueError):runner.load_cases(directory)


def test_unsupervised_native_entry_refuses_before_pins_or_imports(monkeypatch,tmp_path):
    p=tmp_path/'request.json';runner.save_new(p,{'nonce':'f'*48,'freeze':{}})
    monkeypatch.delenv(runner.NONCE_ENV,raising=False)
    monkeypatch.setattr(runner,'_profile_sources',lambda:(_ for _ in ()).throw(AssertionError('must not hash')))
    with pytest.raises(ValueError,match='SUPERVISED'):runner._generate_native(p)


def test_evaluation_retains_three_cold_calls_and_receipts_before_oracles(prepared,monkeypatch,tmp_path):
    directory,generation,_=generated(prepared);record=tmp_path/'evaluation.jsonl'
    frozen=runner.freeze_evaluation(directory,generation,record);calls=mock_evaluation(monkeypatch,frozen)
    ledger=runner.evaluate(frozen,directory,record,generation,'mock','mock')
    assert ledger['status']=='COMPLETE_SYNTHETIC_FINITE_WIDTH_BENEFIT'
    assert calls==[c['check_id'] for c in runner.CHECKS]
    assert len(ledger['checks'])==3 and all(r['passed'] for r in ledger['checks'])
    assert ledger['checks'][1]['observed']['flat_area_mm2']==pytest.approx(1957.6637061435918,abs=1e-10)
    for row in ledger['checks']:
        raw=Path(row['receipt_path']).read_bytes();assert runner._hash(raw)==row['receipt_sha256']
    assert [json.loads(line)['event'] for line in record.read_bytes().splitlines()]==['START','CHECK','CHECK','CHECK','TERMINAL']


def test_prior_oracle_failure_does_not_omit_later_checks(prepared,monkeypatch,tmp_path):
    directory,generation,_=generated(prepared);record=tmp_path/'evaluation.jsonl'
    frozen=runner.freeze_evaluation(directory,generation,record);calls=mock_evaluation(monkeypatch,frozen)
    original=runner.assert_result
    def check(c,case,result,bundle):
        if c['check_id']==runner.CHECKS[0]['check_id']:raise ValueError('AUTHORED_CONTROL_FAILURE')
        return original(c,case,result,bundle)
    monkeypatch.setattr(runner,'assert_result',check)
    ledger=runner.evaluate(frozen,directory,record,generation,'mock','mock')
    assert ledger['status']=='FAILED' and len(calls)==3
    assert [r['passed'] for r in ledger['checks']]==[False,True,True]
    assert all('receipt_path' in r for r in ledger['checks'])


@pytest.mark.parametrize('now,admit',[(360.,True),(360.000001,False)])
def test_immediate_reserved_sixty_seconds_before_every_call(monkeypatch,now,admit):
    monkeypatch.setattr(runner.time,'monotonic',lambda:now)
    monkeypatch.setattr(runner,'_preflight_monitor',lambda:None)
    calls=[];monkeypatch.setattr(runner.legacy_worker,'build_unfold_preview_bounded',lambda *a:calls.append(a))
    case={**runner.case_definition(runner.CASE_IDS[0]),'source_sha256':'a'*64};row={}
    if admit:
        runner._call_admitted(0.,runner.CHECKS[0],case,'AUTHORED',row)
        assert len(calls)==1 and row['status']=='RUNNING'
    else:
        with pytest.raises(ValueError,match='ADMISSION'):runner._call_admitted(0.,runner.CHECKS[0],case,'AUTHORED',row)
        assert not calls and not row


@pytest.mark.parametrize('profile',[runner.LEGACY_PROFILE,runner.FINITE_PROFILE])
@pytest.mark.parametrize('mode',['denied','empty','bad','slow'])
def test_each_profile_call_preflights_and_rechecks_remaining_time(monkeypatch,profile,mode):
    clock={'now':0.};monkeypatch.setattr(runner.time,'monotonic',lambda:clock['now'])
    def rows(_):
        if mode=='denied':raise PermissionError('AUTHORED DENIED')
        if mode=='slow':clock['now']=360.000001;return [(os.getpid(),1)]
        return [] if mode=='empty' else [(os.getpid(),True)]
    monkeypatch.setattr(runner.legacy_worker,'_group_rows',rows)
    calls=[]
    for worker in (runner.legacy_worker,runner.finite_worker):
        monkeypatch.setattr(worker,'build_unfold_preview_bounded',lambda *a:calls.append(a))
    check=next(c for c in runner.CHECKS if c['profile']==profile);row={}
    with pytest.raises(ValueError):runner._call_admitted(0.,check,{},'AUTHORED',row)
    assert not calls and not row


def test_monitor_failure_retains_all_three_checks_without_wrappers(prepared,monkeypatch,tmp_path):
    directory,generation,_=generated(prepared);record=tmp_path/'evaluation.jsonl'
    frozen=runner.freeze_evaluation(directory,generation,record)
    monkeypatch.setattr(runner.legacy_worker,'_group_rows',lambda *_:[])
    calls=[]
    for worker in (runner.legacy_worker,runner.finite_worker):
        monkeypatch.setattr(worker,'build_unfold_preview_bounded',lambda *a:calls.append(a))
    ledger=runner.evaluate(frozen,directory,record,generation,'mock','mock')
    assert not calls and ledger['status']=='FAILED' and len(ledger['checks'])==3
    assert all(r['status']=='FAILED' and r['reason']=='PROCESS_MONITOR_PREFLIGHT_UNAVAILABLE' for r in ledger['checks'])
    assert [json.loads(line)['event'] for line in record.read_bytes().splitlines()]==['START','CHECK','CHECK','CHECK','TERMINAL']


@pytest.mark.parametrize('reason',['UNFOLD_WORKER_BUSY','CAD_HASH_MISMATCH','CAD_CANDIDATE_AXIS_MISMATCH',
    'UNFOLD_TIME_BUDGET_EXCEEDED','NATIVE_DEVELOPMENT_UNAVAILABLE','OVERLAP_CHECK_BUDGET_EXCEEDED'])
def test_negative_cannot_pass_transport_candidate_budget_or_generic_failure(monkeypatch,reason):
    check=runner.CHECKS[2];case={**runner.case_definition(runner.CASE_IDS[1]),'source_sha256':'a'*64}
    result=chart_receipt(check,case,'c'*64);result['reason']=reason
    monkeypatch.setattr(runner,'validate_finite',lambda *a:None)
    with pytest.raises(ValueError,match='PREDECLARED'):runner.assert_result(check,case,result,'c'*64)


@pytest.mark.parametrize('field,value',[('mechanism_version',4.),('transport_version',True),('cache_hit',True),
    ('source_bundle_sha256','d'*64),('transitive_runtime_closure_verified',True),('limits',{'wall_seconds':1})])
def test_typed_cold_transport_receipt_is_required(monkeypatch,field,value):
    check=runner.CHECKS[1];case={**runner.case_definition(runner.CASE_IDS[0]),'source_sha256':'a'*64}
    result=chart_receipt(check,case,'c'*64);result['diagnostics'][field]=value
    monkeypatch.setattr(runner,'validate_finite',lambda *a:None)
    with pytest.raises(ValueError):runner.assert_result(check,case,result,'c'*64)


def test_candidate_coordinates_are_exact_json_not_python_bool_aliases(monkeypatch):
    check=runner.CHECKS[1];case={**runner.case_definition(runner.CASE_IDS[0]),'source_sha256':'a'*64}
    result=chart_receipt(check,case,'c'*64)
    result['diagnostics']['candidate_binding'][0]['line_start'][2]=False
    monkeypatch.setattr(runner,'validate_finite',lambda *a:None)
    with pytest.raises(ValueError,match='COLD_WRAPPER'):runner.assert_result(check,case,result,'c'*64)


def test_legacy_optional_engine_declaration_cannot_contradict_bound_bundle(monkeypatch):
    check=runner.CHECKS[0];case={**runner.case_definition(runner.CASE_IDS[0]),'source_sha256':'a'*64}
    result=chart_receipt(check,case,'b'*64)
    monkeypatch.setattr(runner,'validate_legacy',lambda *a:None)
    result['diagnostics']['engine_source_sha256']='0'*64
    with pytest.raises(ValueError,match='LEGACY_ENGINE_DECLARATION'):runner.assert_result(check,case,result,'b'*64)


@pytest.mark.parametrize('field',['transitive_runtime_closure_verified','hostile_upload_sandbox_verified'])
@pytest.mark.parametrize('value',[True,0,None])
def test_legacy_cannot_claim_unproved_runtime_or_upload_isolation(monkeypatch,field,value):
    check=runner.CHECKS[0];case={**runner.case_definition(runner.CASE_IDS[0]),'source_sha256':'a'*64}
    result=chart_receipt(check,case,'b'*64);result['diagnostics'][field]=value
    monkeypatch.setattr(runner,'validate_legacy',lambda *a:None)
    with pytest.raises(ValueError,match='COLD_WRAPPER'):runner.assert_result(check,case,result,'b'*64)


def test_outer_radius_is_not_an_acceptable_oracle_substitute(monkeypatch):
    check=runner.CHECKS[1];case={**runner.case_definition(runner.CASE_IDS[0]),'source_sha256':'a'*64}
    result=chart_receipt(check,case,'c'*64)
    result['vertices'][0]-=runner.math.pi/2
    monkeypatch.setattr(runner,'validate_finite',lambda *a:None)
    with pytest.raises(ValueError,match='FIXED_INNER'):runner.assert_result(check,case,result,'c'*64)


@pytest.mark.parametrize('mode',['budget','existing','late'])
def test_failed_receipt_or_late_call_retained_honestly(prepared,monkeypatch,tmp_path,mode):
    directory,generation,_=generated(prepared);record=tmp_path/'evaluation.jsonl'
    frozen=runner.freeze_evaluation(directory,generation,record);mock_evaluation(monkeypatch,frozen)
    if mode=='budget':monkeypatch.setitem(runner.EVAL_LIMITS,'total_receipt_bytes',1);frozen=runner.freeze_evaluation(directory,generation,record)
    elif mode=='existing':record.with_name(record.stem+'-'+runner.CHECKS[0]['check_id']+'.json').write_bytes(b'AUTHORED PRIOR RECEIPT')
    else:
        original=runner._call_admitted
        def late(started,check,case,path,row):
            result=original(started,check,case,path,row);row['actual_call_elapsed_seconds']=61.;return result
        monkeypatch.setattr(runner,'_call_admitted',late)
    ledger=runner.evaluate(frozen,directory,record,generation,'mock','mock')
    assert ledger['status']=='FAILED' and len(ledger['checks'])==3
    assert all(r['status']!='NOT_EXECUTED' for r in ledger['checks'])
    if mode=='late':assert all('receipt_path' in r for r in ledger['checks'])
    if mode=='budget':assert all('observed_payload_sha256' in r for r in ledger['checks']) and ledger['total_receipt_bytes']==0
