#!/usr/bin/env python3
"""p69: DECOMPOSITION OF THE PROBATION -- confidence threshold (tau = 0.8) versus conversion of the durations to seconds. YOLO11m, 16 SSMOT sequences, streammux 1280x720.

Plan written BEFORE the run (project: claude/p69_p70_plan.md). Tracker values frozen; nothing is selected.
Setup as in p55 (probe_moj_p55.py, config_infer_11m.txt, MYTRACKER_FPS = FPS of the sequence, the class excludes matching as in the baseline).
New variants of our tracker:
  sec  = minHitsSec 0.6667 and maxAgeSec 3.3333, threshold off        -> /workspace/log_p69_sec_ss_<n>.json
  gate = confHigh 0.8 (2 matches), durations in frames (20, 100)      -> /workspace/log_p69_gate_ss_<n>.json
Existing from p55: base, prob (threshold + seconds), final.
First of all: prob again on SSMOT 1 must be identical to log_p55_prob_ss_1.json (otherwise STOP).
track.h -> track.h.bak_p69 (always restored to the baseline configuration and compiled); engines -> .p69bak (SHA256 check at the end).
Run (GPU free, about 25 min):  nohup python3 /workspace/p69_rastav_probacije.py > /workspace/p69_out.txt 2>&1 &
Monitoring: tail -3 /workspace/p69_out.txt     Evaluation only: python3 /workspace/p69_rastav_probacije.py --eval
If interrupted: python3 /workspace/p69_rastav_probacije.py --vrati   (then rerun -- it resumes)
Output: /workspace/scpilot/p69_rastav_probacije.txt (+ scpilot/p69/p69_sve.json)
"""
import os, re, sys, json, time, shutil, hashlib, subprocess, atexit
import numpy as np
for a, b in (('float', float), ('int', int), ('bool', bool)):
    if not hasattr(np, a): setattr(np, a, b)
W = os.environ.get('P69_W', '/workspace'); S = W + '/scpilot'; O = S + '/p69'; LOCK = O + '/p69.lock'
SRC = W + '/mytracker/src/track.h'; BAK = SRC + '.bak_p69'; LIB = W + '/mytracker/libmoj.so'
PM = W + '/probe_moj_p55.py'; CFG_OWN = W + '/config_tracker_NvSORT.yml'; INFER = W + '/DeepStream-Yolo/config_infer_11m.txt'
ENG11 = W + '/det_glavni/B_yolo11m/weights/best_11m_b1_gpu0_fp16.engine'
PROT0 = [W + '/model_b1_gpu0_fp16.engine', W + '/best_0904.engine', W + '/osnet_vessel.engine', ENG11]
SS16 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
VID = W + '/SSMOT_video/video/SSMOT_%d.mp4'; GT = W + '/SSMOT_video/ground-truth/SSMOT_%d/gt/gt.txt'
BASE = {'maxAge': 100, 'wMeas': 0.15, 'galleryMax': 5, 'wReid': 0.8, 'maxCost': 0.75, 'inertia': 0.10, 'detNms': 0.70, 'minHits': 20}
EXTRA_BASE = {'reidAfterN': 0}
OFF = {'classGate': lambda v: v == 1, 'confHigh': lambda v: v > 1, 'maxAgeSec': lambda v: v <= 0, 'minHitsSec': lambda v: v <= 0}
PROB = {'confHigh': 0.8, 'maxAgeSec': 3.3333, 'minHitsSec': 0.6667}                   # exactly as the p55 'prob'
SEC = {'maxAgeSec': 3.3333, 'minHitsSec': 0.6667}
GATE = {'confHigh': 0.8}
assert dict(SEC, **GATE) == PROB
NEW = [('sec', SEC, 'seconds only (no threshold)'), ('gate', GATE, 'threshold only (durations in frames)')]
CHECK = ('prob', PROB, 1)                                                              # reproducibility check against p55
OLD = {'base': W + '/log_p55_base_ss_%d.json', 'prob': W + '/log_p55_prob_ss_%d.json', 'final': W + '/log_p55_final_ss_%d.json'}
NAME = {'base': 'baseline', 'sec': 'seconds only', 'gate': 'threshold only', 'prob': 'probation only (threshold + seconds)', 'final': 'final variant'}
def outp(k, s): return W + '/log_p69_%s_ss_%d.json' % (k, s)
LOCKS = [S + '/p%d/p%d.lock' % (i, i) for i in (44, 52, 53, 54, 55, 56, 60, 64, 65, 66, 67, 70)]
T_RUN = 900
L = []
def P(x=''): print(x, flush=True); L.append(x)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def json_ok(p):
    try:
        with open(p) as f: d = json.load(f)
        return isinstance(d, list) and len(d) > 0
    except Exception: return False
def alive(lock):
    if not os.path.exists(lock): return None
    pid = open(lock).read().strip()
    return pid if pid and os.path.exists('/proc/' + pid) else None
def stat(p): return (os.stat(p).st_size, os.stat(p).st_mtime_ns) if os.path.exists(p) else None
def uzmi_lock():
    os.makedirs(O, exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
            atexit.register(pusti_lock); return
        except FileExistsError:
            try: pid = open(LOCK).read().strip(); age = time.time() - os.path.getmtime(LOCK)
            except FileNotFoundError: continue
            if (pid and os.path.exists('/proc/' + pid)) or (not pid and age < 10):
                sys.exit('!! p69 is already running (PID %s) -- do NOT run twice.' % (pid or '?'))
            try: os.remove(LOCK)
            except FileNotFoundError: pass
    sys.exit('!! cannot take the lock %s' % LOCK)
def pusti_lock():
    try:
        if open(LOCK).read().strip() == str(os.getpid()): os.remove(LOCK)
    except Exception: pass
def fps(path):
    try:
        import cv2; f = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS)
        if f and f > 0: return float(f)
    except Exception: pass
    return 25.0
def frames(p):
    out = {}
    for fd in json.load(open(p)):
        d = fd.get('detections') or []
        if d: out[int(fd['frame'])] = sorted((int(x['id']), tuple(round(float(v), 2) for v in x['box'][:4])) for x in d)
    return out
# ======================================================================= track.h (same as p52/p53/p55)
RX = lambda n: re.compile(r'^(\s*(int|float|double|bool)\s+' + re.escape(n) + r'\s*=\s*)([^;]+)(;)', re.M)
def getv(txt, n):
    m = list(RX(n).finditer(txt))
    if len(m) != 1: return None, None
    return m[0].group(2), float(m[0].group(3).strip().rstrip('f'))
def setv(txt, n, v):
    m = list(RX(n).finditer(txt)); assert len(m) == 1, n
    ty = m[0].group(2); old = m[0].group(3).strip()
    val = ('%d' % int(round(v))) if ty in ('int', 'bool') else ('%g' % v) + ('f' if old.endswith('f') else '')
    if ty not in ('int', 'bool') and '.' not in val and 'e' not in val: val = val.replace('f', '') + ('.0f' if old.endswith('f') else '.0')
    return txt[:m[0].start(3)] + val + txt[m[0].end(3):]
def check_base(txt):
    bad = []
    for n, v in list(BASE.items()) + list(EXTRA_BASE.items()):
        ty, x = getv(txt, n)
        if x is None: bad.append('%s not found exactly once' % n)
        elif abs(x - v) > 1e-6: bad.append('%s = %g (should be %g)' % (n, x, v))
    for n, okf in OFF.items():
        ty, x = getv(txt, n)
        if x is None: bad.append('%s not found exactly once' % n)
        elif not okf(x): bad.append('%s = %g (should be switched off)' % (n, x))
    return bad
def build():
    t0 = time.time()
    r = subprocess.run('cd %s/mytracker && make clean >/dev/null 2>&1; make 2>&1' % W, shell=True, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(LIB) or os.path.getmtime(LIB) < t0 - 2:
        raise RuntimeError('compilation failed:\n' + (r.stdout or '')[-1500:] + (r.stderr or '')[-1500:])
def vrati():
    if os.path.exists(BAK):
        shutil.copy(BAK, SRC)
        try: build()
        except Exception as e: print('!! compilation of the baseline track.h failed: %s' % e, flush=True)
        if open(SRC).read() == open(BAK).read(): os.remove(BAK)
        print('   track.h restored to the baseline configuration and compiled.', flush=True)
    for p in PROT0:
        b = p + '.p69bak'
        if not os.path.exists(b): continue
        if os.path.exists(p) and sha(p) == sha(b): os.remove(b); print('   %s: untouched, copy deleted' % os.path.basename(p), flush=True)
        else:
            if os.path.exists(p): os.replace(p, p + '.p69_nepoznat')
            os.replace(b, p); print('   %s: RESTORED from the backup copy' % os.path.basename(p), flush=True)
# ======================================================================= GPU
def runs():
    try:
        import scipy, trackeval, cv2   # noqa: F401
    except Exception as e: sys.exit('!! missing package (%s).' % e)
    uzmi_lock()
    for lk in LOCKS:
        if alive(lk): sys.exit('!! %s is still running (PID %s) -- run p69 when it finishes.' % (os.path.basename(lk).split('.')[0], alive(lk)))
    try:
        used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
        print('GPU memory in use: %d MiB' % used, flush=True)
        if used > 1500: sys.exit('GPU is busy (>1500 MiB) -- wait until whatever is running finishes.')
    except (IndexError, ValueError, FileNotFoundError): print('warning: nvidia-smi gave no data, continuing', flush=True)
    if os.path.exists(BAK) or any(os.path.exists(p + '.p69bak') for p in PROT0):
        print('the previous run was interrupted -- restoring track.h and checking the engines:', flush=True); vrati()
    need = [PM, CFG_OWN, INFER, ENG11, SRC, W + '/id_smooth.py'] + [VID % s for s in SS16] + [GT % s for s in SS16] + [OLD[k] % s for k in OLD for s in SS16]
    miss = [p for p in need if not os.path.exists(p)]
    if miss: sys.exit('!! missing (p55 must be done):\n   ' + '\n   '.join(miss))
    t = open(PM).read()
    if not (re.search(r"""set_property\(\s*['"]width['"]\s*,\s*1280\s*\)""", t) and re.search(r"""set_property\(\s*['"]height['"]\s*,\s*720\s*\)""", t)):
        sys.exit('!! probe_moj_p55.py does not have streammux 1280x720 -- tell Claude')
    orig = open(SRC).read(); bad = check_base(orig)
    if bad: sys.exit('!! track.h is NOT at the baseline configuration -- nothing was run:\n   ' + '\n   '.join(bad))
    todo = {v: [s for s in SS16 if not json_ok(outp(v, s))] for v, _, _ in NEW}
    chk_out = outp('provjera_prob', CHECK[2])
    n_all = sum(len(x) for x in todo.values()) + (0 if json_ok(chk_out) else 1)
    print('runs remaining: %d (check + 2 variants x 16 SSMOT sequences)' % n_all, flush=True)
    PROT = [p for p in PROT0 if os.path.exists(p)]
    shutil.copy2(SRC, BAK); sh = {}; t0 = time.time(); done = [0]
    try:
        for p in PROT:
            sh[p] = sha(p); shutil.copy2(p, p + '.p69bak')
            if sha(p + '.p69bak') != sh[p]: sys.exit('!! backup copy %s is not valid -- STOP' % p)
        def one(out, s, lab):
            for att in (1, 2):
                st_ = {p: stat(p) for p in PROT}; env = dict(os.environ, MYTRACKER_FPS='%g' % fps(VID % s)); rc = None
                if os.path.exists(out): os.remove(out)
                try:
                    r = subprocess.run(['python3', PM, VID % s, CFG_OWN, out, INFER], cwd=W, capture_output=True, text=True, env=env, timeout=T_RUN)
                    rc = r.returncode; err = (r.stdout or '')[-2500:] + '\n----\n' + (r.stderr or '')[-3500:]
                except subprocess.TimeoutExpired: err = 'time limit exceeded (%d s)' % T_RUN
                ch = [os.path.basename(p) for p in PROT if stat(p) != st_[p]]
                if ch:
                    if os.path.exists(out): os.remove(out)
                    open(O + '/greska.txt', 'w').write(err)
                    sys.exit('!! %s changed during the run (%s, SSMOT_%d) -- send to Claude: tail -40 /workspace/scpilot/p69/greska.txt' % (', '.join(ch), lab, s))
                if json_ok(out):
                    done[0] += 1; el = time.time() - t0
                    print('  [%3d/%d] %-34s SSMOT %-2d ok   (about %.0f min left)' % (done[0], n_all, lab, s, el / done[0] * (n_all - done[0]) / 60), flush=True); return
                if os.path.exists(out): os.remove(out)
                open(O + '/greska.txt', 'w').write('%s SSMOT_%d, return code %s\n' % (lab, s, rc) + err)
            sys.exit('!! %s, SSMOT %d failed in the second attempt too -- send to Claude: tail -40 /workspace/scpilot/p69/greska.txt' % (lab, s))
        def set_variant(cfg, lab):
            txt = orig
            for kk, x in cfg.items(): txt = setv(txt, kk, x)
            open(SRC, 'w').write(txt); build()
            print('ours, %s: tracker compiled (%s)' % (lab, ', '.join('%s %g' % (kk, x) for kk, x in cfg.items())), flush=True)
        # 1. reproducibility check against p55
        if not json_ok(chk_out):
            set_variant(CHECK[1], 'check: probation only')
            one(chk_out, CHECK[2], 'check: probation only')
        a, b = frames(chk_out), frames(OLD['prob'] % CHECK[2])
        if a != b:
            nf = len(set(a) | set(b)); ne = sum(1 for f in set(a) | set(b) if a.get(f) != b.get(f))
            sys.exit('!! CHECK FAILED: probation only on SSMOT %d differs from p55 in %d of %d frames -- STOP, send this output to Claude.' % (CHECK[2], ne, nf))
        print('reproducibility check: probation only on SSMOT %d identical to p55 (%d frames with detections)' % (CHECK[2], len(a)), flush=True)
        # 2. new variants
        for v, cfg, lab in NEW:
            if not todo[v]: continue
            set_variant(cfg, lab)
            for s in todo[v]: one(outp(v, s), s, 'ours, ' + lab)
    finally:
        if os.path.exists(BAK): shutil.copy(BAK, SRC)
        try: build(); okb = True
        except Exception as e: okb = False; print('!! compilation of the baseline track.h failed: %s' % e, flush=True)
        if okb and os.path.exists(BAK) and open(SRC).read() == open(BAK).read(): os.remove(BAK)
        print('track.h restored to the baseline configuration%s.' % (' and compiled' if okb else ' (COMPILATION FAILED -- tell Claude)'), flush=True)
        for p, hs in sh.items():
            b = p + '.p69bak'
            if os.path.exists(p) and sha(p) == hs:
                if os.path.exists(b): os.remove(b)
            elif os.path.exists(b):
                if os.path.exists(p): os.replace(p, p + '.p69_nepoznat')
                os.replace(b, p); print('   !! %s changed -- RESTORED from the backup copy; tell Claude' % os.path.basename(p), flush=True)
        print('   engines checked (SHA256).', flush=True)
        pusti_lock()
    print('GPU part done (%.0f min).' % ((time.time() - t0) / 60), flush=True)
# ======================================================================= evaluation
def ocjena():
    import trackeval
    from scipy.stats import wilcoxon
    os.makedirs(O, exist_ok=True)
    H = {}; exec(open(W + '/id_smooth.py').read().split('CFG={')[0].split('"""')[2], H)
    load_tr, load_gt, pick = H['load_tr'], H['load_gt'], H['pick']
    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update(PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False, TIME_PROGRESS=False, USE_PARALLEL=False, DISPLAY_LESS_PROGRESS=True)
    rows = lambda d: sorted((int(f), int(o['id']), *map(float, o['box'][:4])) for f, Lst in d.items() for o in Lst)
    def ev(files):
        base = O + '/te_tmp'; shutil.rmtree(base, ignore_errors=True); si = {}
        for s, p in files.items():
            tr = load_tr(p); g = load_gt(GT % s, pick(GT % s, tr))
            G = rows(g); R_ = rows(tr); seq = 'S%02d' % s; si[seq] = max([r[0] for r in G + R_] + [1])
            os.makedirs('%s/gt/%s/gt' % (base, seq), exist_ok=True); os.makedirs(base + '/tr/t/data', exist_ok=True)
            with open('%s/gt/%s/gt/gt.txt' % (base, seq), 'w') as f:
                for x in G: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,1,1\n' % x)
            with open('%s/tr/t/data/%s.txt' % (base, seq), 'w') as f:
                for x in R_: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n' % x)
        dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dc.update(GT_FOLDER=base + '/gt', TRACKERS_FOLDER=base + '/tr', BENCHMARK='b', SPLIT_TO_EVAL='', SEQ_INFO=si, TRACKERS_TO_EVAL=['t'], DO_PREPROC=False,
                  SKIP_SPLIT_FOL=True, GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt', TRACKER_SUB_FOLDER='data', PRINT_CONFIG=False)
        M = [trackeval.metrics.HOTA({'PRINT_CONFIG': False}), trackeval.metrics.CLEAR({'PRINT_CONFIG': False}), trackeval.metrics.Identity({'PRINT_CONFIG': False})]
        out, _ = trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)], M); Rt = out['MotChallenge2DBox']['t']; shutil.rmtree(base, ignore_errors=True)
        res = {}
        for q in si:
            hh = Rt[q]['pedestrian']; m = lambda k: float(np.mean(hh['HOTA'][k]))
            res[int(q[1:])] = {'HOTA': m('HOTA'), 'DetA': m('DetA'), 'AssA': m('AssA'), 'AssPr': m('AssPr'),
                               'IDF1': float(hh['Identity']['IDF1']), 'IDSW': float(hh['CLEAR']['IDSW'])}
        return res
    FILES = {'base': {s: OLD['base'] % s for s in SS16}, 'sec': {s: outp('sec', s) for s in SS16}, 'gate': {s: outp('gate', s) for s in SS16},
             'prob': {s: OLD['prob'] % s for s in SS16}, 'final': {s: OLD['final'] % s for s in SS16}}
    P('=== p69: decomposition of the probation (threshold tau = 0.8 versus durations in seconds), YOLO11m, 16 SSMOT sequences, streammux 1280x720; plan: claude/p69_p70_plan.md')
    if json_ok(outp('provjera_prob', CHECK[2])):
        same = frames(outp('provjera_prob', CHECK[2])) == frames(OLD['prob'] % CHECK[2])
        P('    reproducibility check (probation only, SSMOT %d, against p55): %s' % (CHECK[2], 'IDENTICAL' if same else 'DIFFERENT -- the results are not comparable'))
    R = {}
    for k, f in FILES.items():
        miss = [s for s, p in f.items() if not json_ok(p)]
        if miss: sys.exit('!! no complete logs for "%s" (%d of 16 missing) -- GPU part first' % (NAME[k], len(miss)))
        R[k] = ev(f); print('   TrackEval %-34s ok' % NAME[k], flush=True)
    MET = ['HOTA', 'DetA', 'AssA', 'AssPr', 'IDF1', 'IDSW']
    fmt = lambda m, x: ('%6.1f' if m == 'IDSW' else '%6.3f') % x
    P('\n=== A) MEAN PER SEQUENCE')
    P('  %-34s ' % '' + '  '.join('%6s' % m for m in MET))
    for k in ['base', 'sec', 'gate', 'prob', 'final']:
        P('  %-34s ' % NAME[k] + '  '.join(fmt(m, np.mean([R[k][s][m] for s in SS16])) for m in MET))
    def wp(d):
        d = np.array(d, float)
        if not np.any(d != 0): return 1.0
        try: return float(wilcoxon(d).pvalue)
        except ValueError: return float('nan')
    def holm(ps):
        o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); run = 0
        for r, i in enumerate(o): run = max(run, min(1, ps[i] * (len(ps) - r))); out[i] = run
        return out
    fp = lambda p: '< 0.001' if p < 0.001 else '%.3f' % p
    dl = lambda a, b, m: [R[b][s][m] - R[a][s][m] for s in SS16]
    OUT = {'mean': {k: {m: float(np.mean([R[k][s][m] for s in SS16])) for m in MET} for k in R}, 'kontrasti': {}}
    MAIN = [('sec', 'prob', 'C1: threshold (prob - sec)'), ('base', 'sec', 'C2: seconds (sec - base)')]
    SIDE = [('base', 'gate', 'threshold with frames (gate - base)'), ('gate', 'prob', 'seconds with threshold (prob - gate)'), ('base', 'prob', 'probation (prob - base)')]
    P('\n=== B) MAIN CONTRASTS (two-sided Wilcoxon, n = 16; H = Holm over the 2 main contrasts, for each metric)')
    for grp, pairs, use_holm in (('glavni', MAIN, True), ('sporedni', SIDE, False)):
        if grp == 'sporedni': P('\n=== C) SECONDARY CONTRASTS (descriptive, without correction)')
        ps = {m: [wp(dl(a, b, m)) for a, b, _ in pairs] for m in MET}
        hs = {m: (holm(ps[m]) if use_holm else [None] * len(pairs)) for m in MET}
        for i, (a, b, lab) in enumerate(pairs):
            cells = []
            for m in MET:
                d = dl(a, b, m); e = float(np.mean(d))
                bolje = sum(1 for x in d if (x < 0 if m == 'IDSW' else x > 0))
                cells.append(('%+6.2f' if m == 'IDSW' else '%+.4f') % e + ' p=%s' % fp(ps[m][i]) + (' H=%s' % fp(hs[m][i]) if hs[m][i] is not None else '') + ' %d/16' % bolje)
                OUT['kontrasti'].setdefault(lab, {})[m] = {'d': e, 'p': ps[m][i], 'holm': hs[m][i], 'bolje': bolje}
            P('  %-32s %s' % (lab, ' | '.join(cells)))
    c1 = OUT['kontrasti']['C1: threshold (prob - sec)']['HOTA']; c2 = OUT['kontrasti']['C2: seconds (sec - base)']['HOTA']
    P('\n=== D) RULE FROM THE PLAN (HOTA)')
    P('  C1 = %+.4f (Holm %s), C2 = %+.4f (Holm %s)' % (c1['d'], fp(c1['holm']), c2['d'], fp(c2['holm'])))
    if c1['holm'] < 0.05 and abs(c2['d']) <= 0.005:
        ishod = '1: the gain comes from the threshold -- report C2 and C1 in V-C; the limitation about the non-separated ablation is deleted'
    elif c2['holm'] < 0.05 and abs(c2['d']) > 0.005:
        ishod = '2: the conversion to seconds has its own share -- report both shares, soften the explanation in V-C and VI-A'
    else:
        ishod = '3: no condition met -- report descriptively, keep the limitation'
    P('  outcome %s' % ishod); OUT['ishod'] = ishod
    ad = OUT['kontrasti']['threshold with frames (gate - base)']['HOTA']['d'] + c2['d']; tot = OUT['kontrasti']['probation (prob - base)']['HOTA']['d']
    P('  additivity (descriptive): (gate - base) + (sec - base) = %+.4f, versus prob - base = %+.4f' % (ad, tot))
    open(S + '/p69_rastav_probacije.txt', 'w').write('\n'.join(L) + '\n')
    json.dump(OUT, open(O + '/p69_sve.json', 'w'), indent=1)
    print('\nOutput: /workspace/scpilot/p69_rastav_probacije.txt', flush=True)
if __name__ == '__main__':
    if '--vrati' in sys.argv:
        uzmi_lock(); vrati(); print('done.')
    elif '--eval' in sys.argv: ocjena()
    else:
        runs(); ocjena()
