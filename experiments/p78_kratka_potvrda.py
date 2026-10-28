#!/usr/bin/env python3
"""p78: CONFIDENCE-GATED CONFIRMATION versus FIXED SHORT CONFIRMATION OF ALL TRACKS (H3). YOLO11m, 16 SSMOT sequences, streammux 1280x720.

Plan and decision rule written BEFORE the run: docs/preregistration_p78_p80.md (committed before any run).
Tracker values frozen on 12 Oct 2026; nothing is selected. Setup as in p55/p69 (probe_moj_p55.py, config_infer_11m.txt,
MYTRACKER_FPS = FPS of the sequence). Only confHigh changes: 0.0 means that every detection counts as confident, so every
track is confirmed after minHitsHigh = 2 matches (fixed short confirmation for all tracks).
New variants of our tracker:
  k2  = probation only (p55 'prob') with confHigh 0.0  (class gate as the baseline)  -> /workspace/log_p78_k2_ss_<n>.json
  k2k = final (p55 'final') with confHigh 0.0          (with the class penalty)       -> /workspace/log_p78_k2k_ss_<n>.json
Existing from p55: base, prob (gate + seconds), final (gate + seconds + class penalty).
First of all: final again on SSMOT 1 must be identical to log_p55_final_ss_1.json (otherwise STOP).
Primary contrast: final - k2k (HOTA; FP and IDSW with Holm over the two). Rule: see the docstring of decision() and the preregistration.
track.h -> track.h.bak_p78 (always restored to the baseline configuration and compiled); engines -> .p78bak (SHA256 check at the end).
Run (GPU free, about 25 min):  nohup python3 /workspace/p78_kratka_potvrda.py > /workspace/p78_out.txt 2>&1 &
Monitoring: tail -3 /workspace/p78_out.txt     Evaluation only: PYTHONPATH=/workspace/TrackEval python3 /workspace/p78_kratka_potvrda.py --eval
If interrupted: python3 /workspace/p78_kratka_potvrda.py --vrati   (then rerun -- it resumes)
Output: /workspace/scpilot/p78_kratka_potvrda.txt (+ scpilot/p78/p78_sve.json)
"""
import os, re, sys, json, time, shutil, hashlib, subprocess, atexit
import numpy as np
for a, b in (('float', float), ('int', int), ('bool', bool)):
    if not hasattr(np, a): setattr(np, a, b)
W = os.environ.get('P78_W', '/workspace'); S = W + '/scpilot'; O = S + '/p78'; LOCK = O + '/p78.lock'
SRC = W + '/mytracker/src/track.h'; BAK = SRC + '.bak_p78'; LIB = W + '/mytracker/libmoj.so'
PM = W + '/probe_moj_p55.py'; CFG_OWN = W + '/config_tracker_NvSORT.yml'; INFER = W + '/DeepStream-Yolo/config_infer_11m.txt'
ENG11 = W + '/det_glavni/B_yolo11m/weights/best_11m_b1_gpu0_fp16.engine'
PROT0 = [W + '/model_b1_gpu0_fp16.engine', W + '/best_0904.engine', W + '/osnet_vessel.engine', ENG11]
SS16 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
VID = W + '/SSMOT_video/video/SSMOT_%d.mp4'; GT = W + '/SSMOT_video/ground-truth/SSMOT_%d/gt/gt.txt'
BASE = {'maxAge': 100, 'wMeas': 0.15, 'galleryMax': 5, 'wReid': 0.8, 'maxCost': 0.75, 'inertia': 0.10, 'detNms': 0.70, 'minHits': 20}
EXTRA_BASE = {'reidAfterN': 0}
OFF = {'classGate': lambda v: v == 1, 'confHigh': lambda v: v > 1, 'maxAgeSec': lambda v: v <= 0, 'minHitsSec': lambda v: v <= 0}
PROB = {'confHigh': 0.8, 'maxAgeSec': 3.3333, 'minHitsSec': 0.6667}                   # exactly as the p55 'prob'
FINAL = {'classGate': 0, 'classPenalty': 0.5, 'confHigh': 0.8, 'maxAgeSec': 3.3333, 'minHitsSec': 0.6667}   # exactly as the p55 'final'
K2 = dict(PROB, confHigh=0.0)                                                          # every track confirmed after 2 matches
K2K = dict(FINAL, confHigh=0.0)
NEW = [('k2', K2, 'short confirmation for all (no class penalty)'), ('k2k', K2K, 'short confirmation for all + class penalty')]
CHECK = ('final', FINAL, 1)                                                            # reproducibility check against p55
OLD = {'base': W + '/log_p55_base_ss_%d.json', 'prob': W + '/log_p55_prob_ss_%d.json', 'final': W + '/log_p55_final_ss_%d.json'}
NAME = {'base': 'baseline (20 frames for all)', 'k2': 'short for all (2 matches)', 'prob': 'gated (no class penalty)',
        'k2k': 'short for all + class penalty', 'final': 'final variant (gated + penalty)'}
DELTA = 0.005                                                                          # negligible HOTA difference (as p52, p69)
def outp(k, s): return W + '/log_p78_%s_ss_%d.json' % (k, s)
LOCKS = [S + '/p%d/p%d.lock' % (i, i) for i in (44, 52, 53, 54, 55, 56, 60, 64, 65, 66, 67, 69, 70, 75, 77)]
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
                sys.exit('!! p78 is already running (PID %s) -- do NOT run twice.' % (pid or '?'))
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
        b = p + '.p78bak'
        if not os.path.exists(b): continue
        if os.path.exists(p) and sha(p) == sha(b): os.remove(b); print('   %s: untouched, copy deleted' % os.path.basename(p), flush=True)
        else:
            if os.path.exists(p): os.replace(p, p + '.p78_nepoznat')
            os.replace(b, p); print('   %s: RESTORED from the backup copy' % os.path.basename(p), flush=True)
# ======================================================================= GPU
def runs():
    try:
        import scipy, trackeval, cv2   # noqa: F401
    except Exception as e: sys.exit('!! missing package (%s).' % e)
    uzmi_lock()
    for lk in LOCKS:
        if alive(lk): sys.exit('!! %s is still running (PID %s) -- run p78 when it finishes.' % (os.path.basename(lk).split('.')[0], alive(lk)))
    try:
        used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
        print('GPU memory in use: %d MiB' % used, flush=True)
        if used > 1500: sys.exit('GPU is busy (>1500 MiB) -- wait until whatever is running finishes.')
    except (IndexError, ValueError, FileNotFoundError): print('warning: nvidia-smi gave no data, continuing', flush=True)
    if os.path.exists(BAK) or any(os.path.exists(p + '.p78bak') for p in PROT0):
        print('the previous run was interrupted -- restoring track.h and checking the engines:', flush=True); vrati()
    need = [PM, CFG_OWN, INFER, ENG11, SRC, W + '/id_smooth.py'] + [VID % s for s in SS16] + [GT % s for s in SS16] + [OLD[k] % s for k in OLD for s in SS16]
    miss = [p for p in need if not os.path.exists(p)]
    if miss: sys.exit('!! missing (p55 must be done):\n   ' + '\n   '.join(miss))
    t = open(PM).read()
    if not (re.search(r"""set_property\(\s*['"]width['"]\s*,\s*1280\s*\)""", t) and re.search(r"""set_property\(\s*['"]height['"]\s*,\s*720\s*\)""", t)):
        sys.exit('!! probe_moj_p55.py does not have streammux 1280x720 -- report this')
    orig = open(SRC).read(); bad = check_base(orig)
    if bad: sys.exit('!! track.h is NOT at the baseline configuration -- nothing was run:\n   ' + '\n   '.join(bad))
    todo = {v: [s for s in SS16 if not json_ok(outp(v, s))] for v, _, _ in NEW}
    chk_out = outp('provjera_final', CHECK[2])
    n_all = sum(len(x) for x in todo.values()) + (0 if json_ok(chk_out) else 1)
    print('runs remaining: %d (check + 2 variants x 16 SSMOT sequences)' % n_all, flush=True)
    PROT = [p for p in PROT0 if os.path.exists(p)]
    shutil.copy2(SRC, BAK); sh = {}; t0 = time.time(); done = [0]
    try:
        for p in PROT:
            sh[p] = sha(p); shutil.copy2(p, p + '.p78bak')
            if sha(p + '.p78bak') != sh[p]: sys.exit('!! backup copy %s is not valid -- STOP' % p)
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
                    sys.exit('!! %s changed during the run (%s, SSMOT_%d) -- see: tail -40 /workspace/scpilot/p78/greska.txt' % (', '.join(ch), lab, s))
                if json_ok(out):
                    done[0] += 1; el = time.time() - t0
                    print('  [%3d/%d] %-34s SSMOT %-2d ok   (about %.0f min left)' % (done[0], n_all, lab, s, el / done[0] * (n_all - done[0]) / 60), flush=True); return
                if os.path.exists(out): os.remove(out)
                open(O + '/greska.txt', 'w').write('%s SSMOT_%d, return code %s\n' % (lab, s, rc) + err)
            sys.exit('!! %s, SSMOT %d failed in the second attempt too -- see: tail -40 /workspace/scpilot/p78/greska.txt' % (lab, s))
        def set_variant(cfg, lab):
            txt = orig
            for kk, x in cfg.items(): txt = setv(txt, kk, x)
            open(SRC, 'w').write(txt); build()
            print('ours, %s: tracker compiled (%s)' % (lab, ', '.join('%s %g' % (kk, x) for kk, x in cfg.items())), flush=True)
        # 1. reproducibility check against p55
        if not json_ok(chk_out):
            set_variant(CHECK[1], 'check: final variant')
            one(chk_out, CHECK[2], 'check: final variant')
        a, b = frames(chk_out), frames(OLD['final'] % CHECK[2])
        if a != b:
            nf = len(set(a) | set(b)); ne = sum(1 for f in set(a) | set(b) if a.get(f) != b.get(f))
            sys.exit('!! CHECK FAILED: final variant on SSMOT %d differs from p55 in %d of %d frames -- STOP, report this output.' % (CHECK[2], ne, nf))
        print('reproducibility check: final variant on SSMOT %d identical to p55 (%d frames with detections)' % (CHECK[2], len(a)), flush=True)
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
        print('track.h restored to the baseline configuration%s.' % (' and compiled' if okb else ' (COMPILATION FAILED -- report this)'), flush=True)
        for p, hs in sh.items():
            b = p + '.p78bak'
            if os.path.exists(p) and sha(p) == hs:
                if os.path.exists(b): os.remove(b)
            elif os.path.exists(b):
                if os.path.exists(p): os.replace(p, p + '.p78_nepoznat')
                os.replace(b, p); print('   !! %s changed -- RESTORED from the backup copy; report this' % os.path.basename(p), flush=True)
        print('   engines checked (SHA256).', flush=True)
        pusti_lock()
    print('GPU part done (%.0f min).' % ((time.time() - t0) / 60), flush=True)
# ======================================================================= evaluation
def wp(d):
    from scipy.stats import wilcoxon
    d = np.array(d, float)
    if not np.any(d != 0): return 1.0
    try: return float(wilcoxon(d).pvalue)
    except ValueError: return float('nan')
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); run = 0
    for r, i in enumerate(o): run = max(run, min(1, ps[i] * (len(ps) - r))); out[i] = run
    return out
def decision(dhota, p_hota, d_err, p_err_holm):
    """Rule from docs/preregistration_p78_p80.md (primary contrast final - k2k, delta = 0.005 HOTA).
    A+: mean dHOTA > +delta and Wilcoxon p(HOTA) < 0.05 -> the gate is more accurate than short confirmation for all;
        H3 holds (also in the stronger form: more accurate than both fixed confirmations).
    A:  mean dHOTA >= -delta and, for FP or IDSW, final lower than k2k with Holm-adjusted p < 0.05 (Holm over the two)
        -> H3 holds as worded (gated confirmation: no less accurate than short confirmation for all, fewer errors).
    B:  mean dHOTA < -delta and Wilcoxon p(HOTA) < 0.05 -> short confirmation for all is more accurate; H3 reworded.
    C:  otherwise -> the gate has no measurable own effect beside short confirmation; H3 reworded as short confirmation."""
    fewer = [m for m in ('FP', 'IDSW') if d_err[m] < 0 and p_err_holm[m] < 0.05]
    if dhota > DELTA and p_hota < 0.05:
        return 'A+', 'gated confirmation more accurate than short confirmation for all (dHOTA %+.4f, p %.3f)%s: H3 holds' % (
            dhota, p_hota, ('; also fewer ' + ' and '.join(fewer)) if fewer else '')
    if dhota >= -DELTA and fewer:
        return 'A', 'H3 holds as worded: dHOTA %+.4f >= -%.3f and fewer %s (Holm < 0.05)' % (dhota, DELTA, ' and '.join(fewer))
    if dhota < -DELTA and p_hota < 0.05:
        return 'B', 'short confirmation for all is more accurate (dHOTA %+.4f, p %.3f): H3 reworded, the gate has no advantage' % (dhota, p_hota)
    return 'C', 'no measurable own effect of the gate beside short confirmation (dHOTA %+.4f, p %.3f; FP/IDSW not significantly lower): H3 reworded' % (dhota, p_hota)
def ocjena():
    import trackeval
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
            res[int(q[1:])] = {'HOTA': m('HOTA'), 'DetA': m('DetA'), 'AssA': m('AssA'), 'IDF1': float(hh['Identity']['IDF1']),
                               'IDSW': float(hh['CLEAR']['IDSW']), 'FP': float(hh['CLEAR']['CLR_FP']), 'FN': float(hh['CLEAR']['CLR_FN']),
                               'Frag': float(hh['CLEAR']['Frag'])}
        return res
    FILES = {'base': {s: OLD['base'] % s for s in SS16}, 'k2': {s: outp('k2', s) for s in SS16}, 'prob': {s: OLD['prob'] % s for s in SS16},
             'k2k': {s: outp('k2k', s) for s in SS16}, 'final': {s: OLD['final'] % s for s in SS16}}
    P('=== p78: confidence-gated confirmation versus fixed short confirmation of all tracks (H3), YOLO11m, 16 SSMOT sequences, streammux 1280x720')
    P('    rule written before the run: docs/preregistration_p78_p80.md; delta = %.3f HOTA' % DELTA)
    if json_ok(outp('provjera_final', CHECK[2])):
        same = frames(outp('provjera_final', CHECK[2])) == frames(OLD['final'] % CHECK[2])
        P('    reproducibility check (final variant, SSMOT %d, against p55): %s' % (CHECK[2], 'IDENTICAL' if same else 'DIFFERENT -- the results are not comparable'))
    R = {}
    for k, f in FILES.items():
        miss = [s for s, p in f.items() if not json_ok(p)]
        if miss: sys.exit('!! no complete logs for "%s" (%d of 16 missing) -- GPU part first' % (NAME[k], len(miss)))
        R[k] = ev(f); print('   TrackEval %-34s ok' % NAME[k], flush=True)
    MET = ['HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW', 'FP', 'FN', 'Frag']
    INT = ('IDSW', 'FP', 'FN', 'Frag')
    fmt = lambda m, x: ('%7.1f' if m in INT else '%7.3f') % x
    P('\n=== A) MEAN PER SEQUENCE')
    P('  %-36s ' % '' + ' '.join('%7s' % m for m in MET))
    for k in ['base', 'k2', 'prob', 'k2k', 'final']:
        P('  %-36s ' % NAME[k] + ' '.join(fmt(m, np.mean([R[k][s][m] for s in SS16])) for m in MET))
    fp = lambda p: '< 0.001' if p < 0.001 else '%.3f' % p
    dl = lambda a, b, m: [R[b][s][m] - R[a][s][m] for s in SS16]
    OUT = {'mean': {k: {m: float(np.mean([R[k][s][m] for s in SS16])) for m in MET} for k in R}, 'kontrasti': {}}
    def row(a, b, lab):
        cells = []
        for m in MET:
            d = dl(a, b, m); e = float(np.mean(d)); p = wp(d)
            better = sum(1 for x in d if (x < 0 if m in INT else x > 0))
            cells.append(('%+6.2f' if m in INT else '%+.4f') % e + ' p=%s %d/16' % (fp(p), better))
            OUT['kontrasti'].setdefault(lab, {})[m] = {'d': e, 'p': p, 'better': better}
        P('  %-30s %s' % (lab, ' | '.join(cells)))
    P('\n=== B) PRIMARY CONTRAST: final - k2k (two-sided Wilcoxon, n = 16; "better" = sequences where final is better)')
    row('k2k', 'final', 'final - k2k')
    d_err = {m: OUT['kontrasti']['final - k2k'][m]['d'] for m in ('FP', 'IDSW')}
    hp = holm([OUT['kontrasti']['final - k2k'][m]['p'] for m in ('FP', 'IDSW')]); p_err_holm = {'FP': hp[0], 'IDSW': hp[1]}
    P('  Holm over FP and IDSW: FP %s, IDSW %s' % (fp(hp[0]), fp(hp[1])))
    hh = OUT['kontrasti']['final - k2k']['HOTA']
    code, txt = decision(hh['d'], hh['p'], d_err, p_err_holm)
    P('\n=== C) RULE FROM THE PREREGISTRATION')
    P('  outcome %s: %s' % (code, txt)); OUT['ishod'] = {'code': code, 'text': txt, 'holm_FP': hp[0], 'holm_IDSW': hp[1]}
    P('\n=== D) SECONDARY CONTRASTS (descriptive, without correction)')
    row('k2', 'prob', 'gated - short (no penalty)')
    row('base', 'k2', 'short - long (no penalty)')
    row('base', 'prob', 'gated - long (no penalty)')
    open(S + '/p78_kratka_potvrda.txt', 'w').write('\n'.join(L) + '\n')
    json.dump(OUT, open(O + '/p78_sve.json', 'w'), indent=1)
    print('\nOutput: /workspace/scpilot/p78_kratka_potvrda.txt', flush=True)
if __name__ == '__main__':
    if '--vrati' in sys.argv:
        uzmi_lock(); vrati(); print('done.')
    elif '--eval' in sys.argv: ocjena()
    else:
        runs(); ocjena()
