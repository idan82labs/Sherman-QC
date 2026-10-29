#!/usr/bin/env python3
"""Checkpoint running Sherman workflows: save every finished agent result from their
journals to disk and generate continuation scripts that replay those results and run
only the missing agents. Safe to run any time (idempotent). Usage: python3 make_cont.py"""
import json, os, re, sys
WF = '/Users/idantal/.claude/projects/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/subagents/workflows'
SC = '/Users/idantal/.claude/projects/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/workflows/scripts'
CONT = '/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/cont'
OUT = os.path.dirname(os.path.abspath(__file__))
JOBS = {
  # name: (unwrapped source script, [journals oldest->newest], marker line where the body starts)
  'wave4': (f'{OUT}/src/wave4.js', ['wf_6cb88506-c77', 'wf_bcbaf368-ceb', 'wf_d8545cc3-b34', 'wf_690b4b55-a31', 'wf_2c416935-c89'], 'const jevP = agent('),
  'g3fix': (f'{OUT}/src/g3fix.js', ['wf_ac66546b-4b6', 'wf_97773438-a0c'], "phase('List')"),
  'gold_v2': (f'{OUT}/src/gold_v2.js', ['wf_80a10842-890', 'wf_edc55c38-ded', 'wf_28c9efdc-4e3'], "phase('Select')"),
}
safe = lambda s: re.sub(r'[^A-Za-z0-9_.-]', '_', s)
report = {}
for name, (src, journals, marker) in JOBS.items():
    done, inflight = {}, {}
    for j in journals:
        p = f'{WF}/{j}/journal.jsonl'
        if not os.path.exists(p): continue
        lab = {}
        for l in open(p):
            d = json.loads(l)
            if d['type'] == 'started':
                lab[d['agentId']] = d.get('label'); inflight[d['agentId']] = d.get('label')
            elif d['type'] in ('result', 'failed'):
                inflight.pop(d.get('agentId'), None)
                L = lab.get(d.get('agentId'))
                if d['type'] == 'result' and d.get('result') is not None and L and not L.startswith('load:'):
                    done[L] = d['result']
    D = f'{OUT}/{name}_done'; os.makedirs(D, exist_ok=True)
    for k, v in done.items(): json.dump(v, open(f'{D}/{safe(k)}.json', 'w'), indent=1)
    s = open(src).read(); i = s.index(marker); head, body = s[:i], s[i:]
    body = re.sub(r'\bagent\(', 'A(', body)
    wrap = f'''
const DONE_DIR = '{D}'
const DONE_LABELS = new Set({json.dumps(sorted(done))})
const safeL = (s) => s.replace(/[^A-Za-z0-9_.-]/g, '_')
const A = (p, o) => (o && DONE_LABELS.has(o.label))
  ? (o.schema
      ? agent(`Read the file ${{DONE_DIR}}/${{safeL(o.label)}}.json with cat and return its content EXACTLY as your structured output (every field unchanged). Do nothing else.`, {{ ...o, label: 'load:' + o.label, effort: 'low' }})
      : agent(`Read the file ${{DONE_DIR}}/${{safeL(o.label)}}.json with cat. It holds one JSON string. Output ONLY that string's decoded text, verbatim. Do nothing else.`, {{ ...o, label: 'load:' + o.label, effort: 'low' }}))
  : agent(p, o)
'''
    head = re.sub(r"name: '([^']+)'", lambda m: f"name: '{m.group(1).replace('-continue','')}-continue'", head, count=1)
    open(f'{OUT}/{name}_cont.js', 'w').write(head + wrap + body)
    report[name] = {'done': len(done), 'in_flight_lost': sorted(set(x for x in inflight.values() if x and not x.startswith('load:') and x not in done))}
json.dump(report, open(f'{OUT}/checkpoint_status.json', 'w'), indent=1)
print(json.dumps(report, indent=1))
