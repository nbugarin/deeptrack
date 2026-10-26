#!/usr/bin/env python3
"""p70: HOW MANY TIMES THE EXCEPTION HANDLER RUNS (unfreeze, BoxMOT 10.0.83) FOR OC-SORT, DEEP OC-SORT AND HYBRID-SORT -- 16 SSMOT sequences

Plan written BEFORE the run (project: claude/p69_p70_plan.md).
p56 recorded the number of calls only for Hybrid-SORT (886, logged for 15 of the 16 sequences). This script repeats p56 for the three trackers of the OC-SORT family
with the SAME code (imports /workspace/p56_iste_detekcije.py: the same patch, the same constructors, the same detections scpilot/p54/det/ds_mux_<n>.json,
the same OSNet), counts IndexError and ValueError per sequence and compares the output with the output of p56 (IDs and boxes to 0.01 px).
Does not touch track.h, the engines or the configurations; writes only to scpilot/p70.
Run (GPU free, p69 done; about 25 min):  nohup python3 /workspace/p70_rukovatelj.py > /workspace/p70_out.txt 2>&1 &
Monitoring: tail -3 /workspace/p70_out.txt     Printout only: python3 /workspace/p70_rukovatelj.py --eval
Output: /workspace/scpilot/p70_rukovatelj.txt (+ scpilot/p70/p70_sve.json); a new run resumes (finished sequences are skipped)
"""
import os, sys, json, time, importlib.util, subprocess, atexit
import numpy as np
for a, b in (('float', float), ('int', int), ('bool', bool)):
    if not hasattr(np, a): setattr(np, a, b)
W = os.environ.get('P56_W', '/workspace'); S = W + '/scpilot'; O = S + '/p70'; TR = O + '/trk'; LOCK = O + '/p70.lock'
P56 = W + '/p56_iste_detekcije.py'
NAMES = ['OC-SORT', 'Deep OC-SORT', 'Hybrid-SORT']
LOCKS = [S + '/p%d/p%d.lock' % (i, i) for i in (44, 52, 53, 54, 55, 56, 60, 64, 65, 66, 67, 69)]
L = []
def P(x=''): print(x, flush=True); L.append(x)
def ucitaj_p56():
    if not os.path.exists(P56): sys.exit('!! missing %s' % P56)
    spec = importlib.util.spec_from_file_location('p56', P56); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m
def outp(name, s): return TR + '/%s_%d.json' % (name.replace(' ', '_').replace('-', ''), s)
def json_ok(p):
    try:
        with open(p) as f: d = json.load(f)
        return isinstance(d, list) and len(d) > 0
    except Exception: return False
def frames(p):
    out = {}
    for fd in json.load(open(p)):
        d = fd.get('detections') or []
        if d: out[int(fd['frame'])] = sorted((int(x['id']), tuple(round(float(v), 2) for v in x['box'][:4])) for x in d)
    return out
def alive(lock):
    if not os.path.exists(lock): return None
    pid = open(lock).read().strip()
    return pid if pid and os.path.exists('/proc/' + pid) else None
def uzmi_lock():
    os.makedirs(O, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
    except FileExistsError:
        if alive(LOCK): sys.exit('!! p70 is already running -- do NOT run twice.')
        os.remove(LOCK); return uzmi_lock()
    atexit.register(lambda: os.path.exists(LOCK) and open(LOCK).read().strip() == str(os.getpid()) and os.remove(LOCK))
def stanje():
    try: return json.load(open(O + '/stanje.json'))
    except Exception: return {}
def runs():
    m = ucitaj_p56()
    try:
        import cv2, torch   # noqa: F401
    except Exception as e: sys.exit('!! missing package (%s)' % e)
    uzmi_lock(); os.makedirs(TR, exist_ok=True)
    for lk in LOCKS:
        if alive(lk): sys.exit('!! %s is still running -- run p70 when it finishes.' % os.path.basename(lk).split('.')[0])
    try:
        used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
        print('GPU memory in use: %d MiB' % used, flush=True)
        if used > 1500: sys.exit('GPU is busy (>1500 MiB) -- wait.')
    except (IndexError, ValueError, FileNotFoundError): print('warning: nvidia-smi gave no data, continuing', flush=True)
    miss = [p for p in [m.REID] + [m.DET % s for s in m.SS16] + [m.VID % s for s in m.SS16] + [m.outp(n, s) for n in NAMES for s in m.SS16] if not os.path.exists(p)]
    if miss: sys.exit('!! missing (p54 and p56 must be done):\n   ' + '\n   '.join(miss[:20]))
    import cv2, torch
    boxmot = m.boxmot_setup(); dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('BoxMOT version: %s; device: %s' % (getattr(boxmot, '__version__', '?'), dev), flush=True)
    cls = {}
    for name, cands in m.CAND:
        if name in NAMES:
            c = next((getattr(boxmot, x) for x in cands if hasattr(boxmot, x)), None)
            if c is None: sys.exit('!! %s is not in this version of BoxMOT' % name)
            cls[name] = c
    st = stanje(); st.setdefault('preskok', {}); st.setdefault('vrijeme', {})
    DE = {s: json.load(open(m.DET % s)) for s in m.SS16}; t0 = time.time()
    for name in NAMES:
        for s in m.SS16:
            key = '%s|%d' % (name, s)
            if json_ok(outp(name, s)) and key in st['preskok']: continue
            m.PRESKOK['IndexError'] = 0; m.PRESKOK['ValueError'] = 0
            cap = cv2.VideoCapture(m.VID % s); fr = cap.get(cv2.CAP_PROP_FPS) or 25.0
            kw, used = m.kwargs_for(cls[name], dev, fr); t = cls[name](**kw)
            byf = {int(fd['frame']): fd.get('detections') or [] for fd in DE[s]}; f = 0; T = []; ttr = 0.0
            while True:
                ok, img = cap.read()
                if not ok: break
                f += 1; dl = byf.get(f, [])
                d = np.array([[x['box'][0], x['box'][1], x['box'][0] + x['box'][2], x['box'][1] + x['box'][3], x.get('conf', 1.0), x.get('class', 0)] for x in dl], float) if dl else np.zeros((0, 6))
                ta = time.time(); o = t.update(d, img); ttr += time.time() - ta
                T.append({'frame': f, 'detections': [{'id': int(r[4]), 'box': [float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])]} for r in (o if o is not None else [])]})
            json.dump(T, open(outp(name, s) + '.tmp', 'w')); os.replace(outp(name, s) + '.tmp', outp(name, s))
            st['preskok'][key] = dict(m.PRESKOK); st['vrijeme'][key] = [f, ttr]; json.dump(st, open(O + '/stanje.json', 'w'), indent=1); del t
            print('  %-13s SSMOT %-2d ok  IndexError %4d  ValueError %4d  (%.0f s; total %.0f min)' % (name, s, m.PRESKOK['IndexError'], m.PRESKOK['ValueError'], ttr, (time.time() - t0) / 60), flush=True)
    print('GPU part done (%.0f min).' % ((time.time() - t0) / 60), flush=True)
def ocjena():
    m = ucitaj_p56(); st = stanje(); pk = st.get('preskok', {}); OUT = {}
    P('=== p70: exception handler in unfreeze (BoxMOT 10.0.83), 16 SSMOT sequences, the same detections as p56; plan: claude/p69_p70_plan.md')
    P('  %-13s %10s %10s %18s %22s' % ('tracker', 'IndexError', 'ValueError', 'seqs with a call', 'identical to p56'))
    for name in NAMES:
        v = {s: pk['%s|%d' % (name, s)] for s in m.SS16 if '%s|%d' % (name, s) in pk}
        ie = sum(x['IndexError'] for x in v.values()); ve = sum(x['ValueError'] for x in v.values())
        nz = sum(1 for x in v.values() if x['IndexError'] or x['ValueError'])
        same = 0; diff_fr = {}
        for s in m.SS16:
            if not (json_ok(outp(name, s)) and json_ok(m.outp(name, s))): continue
            a, b = frames(outp(name, s)), frames(m.outp(name, s))
            if a == b: same += 1
            else: diff_fr[s] = [sum(1 for f in set(a) | set(b) if a.get(f) != b.get(f)), len(set(a) | set(b))]
        P('  %-13s %10d %10d %13d / %-2d %17d / 16' % (name, ie, ve, nz, len(v), same))
        for s, (nd, nf) in diff_fr.items(): P('      SSMOT %-2d: %d of %d frames with detections differ' % (s, nd, nf))
        OUT[name] = {'IndexError': ie, 'ValueError': ve, 'snimaka_s_pozivom': nz, 'zabiljezeno': len(v), 'identicno': same, 'razlike': diff_fr,
                     'po_snimci': {str(s): x for s, x in v.items()}}
    P('\n=== RULE FROM THE PLAN')
    sve_isto = all(OUT[n]['identicno'] == 16 for n in NAMES); zab = all(OUT[n]['zabiljezeno'] == 16 for n in NAMES)
    if not zab: P('  not all 16 sequences recorded for all trackers -- first finish the GPU part')
    elif not sve_isto: P('  outcome 3: output NOT identical to p56 -- the numbers in the paper do not change; report the counts as "in a repeated run"; tell Claude')
    else:
        oc = OUT['OC-SORT']['IndexError'] + OUT['OC-SORT']['ValueError']; do = OUT['Deep OC-SORT']['IndexError'] + OUT['Deep OC-SORT']['ValueError']
        if oc == 0 and do == 0: P('  outcome 1: output identical to p56; for OC-SORT and Deep OC-SORT the handler was NOT invoked -> in IV-D "never invoked"')
        else: P('  outcome 2: output identical to p56; handler also invoked for OC-SORT (%d) and/or Deep OC-SORT (%d) -> report in IV-D and VII' % (oc, do))
    open(S + '/p70_rukovatelj.txt', 'w').write('\n'.join(L) + '\n'); json.dump(OUT, open(O + '/p70_sve.json', 'w'), indent=1)
    print('\nOutput: /workspace/scpilot/p70_rukovatelj.txt', flush=True)
if __name__ == '__main__':
    if '--eval' in sys.argv: ocjena()
    else:
        runs(); ocjena()
