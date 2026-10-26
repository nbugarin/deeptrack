#!/usr/bin/env python3
"""p60: FIRST TEST ON ANOTHER SET WITH GROUND TRUTH -- FVessel (Clip-10: 10 sequences, Yangtze river, fixed camera, 2560x1440, 25 fps).

The same pipeline as p55/p59 (YOLO11m, the same yml files, streammux at the native resolution), tracker values FROZEN (final
variant from the IEEE paper); nothing is tuned on FVessel. Trackers: ours final + NvSORT, NvDeepSORT, NvDCF r1, NvDCF r2.
Also: ours + the track confirmation rule from p58 (v1 frozen, v2 = size branch only with confidence < 0.7).

The FVessel ground truth is annotated ONCE PER SECOND (first column = second). Evaluation: the tracker output is taken in the frame that
corresponds to that second, so TrackEval runs on a 1 Hz sequence (HOTA, IDF1, AssA, DetA, IDSW, FP, FN). The alignment second -> frame
(frame = second x fps + k) is chosen ONCE for the whole set so that the GT best matches the detections of ALL trackers together
(the same k for all trackers, so the comparison is fair); k per sequence is also printed as a check.

Run (GPU free):           python3 /workspace/p60_fvessel.py
Re-evaluation only:       PYTHONPATH=/workspace/TrackEval python3 /workspace/p60_fvessel.py --analiza
If interrupted:           python3 /workspace/p60_fvessel.py --vrati     (then rerun -- it resumes)
Output: /workspace/scpilot/p60_fvessel.txt (+ scpilot/p60/)
With the COCO detector (after p60_coco_priprema.py):
  nohup python3 /workspace/p60_fvessel.py /workspace/Clip-10_h264 --det=coco > /workspace/p60c_out.txt 2>&1 &
  -> /workspace/scpilot/p60_fvessel_coco.txt (+ scpilot/p60_coco/)

OFFSET SENSITIVITY (23.10.2026, added for IEEE v21): --pomak=K uses a GIVEN common offset k = K instead of the one that
maximises the match with the GT (that choice uses the same IoU that is evaluated). Only re-evaluation of the existing logs (implies
--analiza, no GPU needed); the main result is NOT overwritten -> scpilot/p60_fvessel[_coco]_pomak<K>.txt (+ rezultati_pomak<K>.json).
  PYTHONPATH=/workspace/TrackEval python3 /workspace/p60_fvessel.py /workspace/Clip-10_h264 --det=coco --pomak=0
"""
import os, re, sys, json, time, glob, shutil, hashlib, subprocess, atexit, warnings
warnings.filterwarnings('ignore')
import numpy as np
for _a, _b in (('float', float), ('int', int), ('bool', bool)):
    if _a not in np.__dict__: setattr(np, _a, _b)

W = os.environ.get('P60_W', '/workspace'); S = W + '/scpilot'; O = S + '/p60'; LOCK = O + '/p60.lock'
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
FV = ARGS[0] if ARGS else W + '/Clip-10'
SRC = W + '/mytracker/src/track.h'; BAK = SRC + '.bak_p60'; LIB = W + '/mytracker/libmoj.so'
PM_SRC = W + '/probe_moj.py'; PT_SRC = W + '/probe_to_json.py'; PM = W + '/probe_moj_p60.py'; PTJ = W + '/probe_to_json_p60.py'
CFG_OWN = W + '/config_tracker_NvSORT.yml'
INFER = W + '/DeepStream-Yolo/config_infer_11m.txt'
ENG11 = W + '/det_glavni/B_yolo11m/weights/best_11m_b1_gpu0_fp16.engine'
PROT0 = [W + '/model_b1_gpu0_fp16.engine', W + '/best_0904.engine', W + '/osnet_vessel.engine', ENG11]
BASE = {'maxAge': 100, 'wMeas': 0.15, 'galleryMax': 5, 'wReid': 0.8, 'maxCost': 0.75, 'inertia': 0.10, 'detNms': 0.70, 'minHits': 20}
EXTRA_BASE = {'reidAfterN': 0}
OFF = {'classGate': lambda v: v == 1, 'confHigh': lambda v: v > 1, 'maxAgeSec': lambda v: v <= 0, 'minHitsSec': lambda v: v <= 0}
FINAL = {'classGate': 0, 'classPenalty': 0.5, 'confHigh': 0.8, 'maxAgeSec': 3.3333, 'minHitsSec': 0.6667}
NV = [('nvsort', W + '/config_tracker_NvSORT.yml', 'NvSORT'), ('deep', W + '/config_tracker_maritime.yml', 'NvDeepSORT'),
      ('dcf1', W + '/cfg_dcf_novis.yml', 'NvDCF r1'), ('dcf2', W + '/cfg_nvdcf.yml', 'NvDCF r2')]
NAZIV = dict([('final', 'ours, final'), ('final_v1', 'ours + rule v1'), ('final_v2', 'ours + rule v2')] +
             [('nv_' + k, lab) for k, _, lab in NV])
PRAVILO = dict(T_MIN=3.0, STATIC=0.5, C_LOW=0.5, A_BIG=0.25)          # frozen in p58
LOCKS = [S + '/p55/p55.lock', S + '/p56/p56.lock', S + '/p59/p59.lock']
# --det=coco: general COCO YOLO11m (class boat only, input 640; prepared with p60_coco_priprema.py) instead of the SSMOT detector.
# Trackers and everything else the same; logs and result separate (scpilot/p60_coco/, scpilot/p60_fvessel_coco.txt).
DET = 'coco' if '--det=coco' in sys.argv else 'ssmot'
if DET == 'coco':
    O = S + '/p60_coco'; LOCK = O + '/p60.lock'
    INFER = W + '/DeepStream-Yolo/config_infer_coco11m_640.txt'
    ENGC = W + '/coco_det/yolo11m_coco_640_b1_gpu0_fp16.engine'
    PROT0 = PROT0 + [ENGC]; LOCKS = LOCKS + [S + '/p60/p60.lock']; REZTXT = S + '/p60_fvessel_coco.txt'
    DET_OPIS = 'general COCO YOLO11m (class boat only, input 640, without training)'
else:
    ENGC = None; LOCKS = LOCKS + [S + '/p60_coco/p60.lock']; REZTXT = S + '/p60_fvessel.txt'
    DET_OPIS = 'YOLO11m trained on SSMOT (input 640)'
# --pomak=K: given common offset (sensitivity); analysis only, separate output
POMAK = None
for _a in sys.argv[1:]:
    if _a.startswith('--pomak='):
        try: POMAK = int(_a.split('=', 1)[1])
        except ValueError: sys.exit('!! --pomak must be a whole number of frames, e.g. --pomak=0 or --pomak=-3')
if POMAK is not None:
    REZTXT = REZTXT[:-4] + '_pomak%+d.txt' % POMAK

L = []
def P(x=''):
    print(x, flush=True); L.append(x)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def json_ok(p):
    try:
        d = json.load(open(p)); return isinstance(d, list) and len(d) > 0
    except Exception: return False
def stat(p): return (os.stat(p).st_size, os.stat(p).st_mtime_ns) if os.path.exists(p) else None
def alive(lock):
    if not os.path.exists(lock): return None
    pid = open(lock).read().strip()
    return pid if pid and os.path.exists('/proc/' + pid) else None
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
                sys.exit('!! p60 is already running (PID %s) -- do NOT run twice.' % (pid or '?'))
            try: os.remove(LOCK)
            except FileNotFoundError: pass
    sys.exit('!! cannot take the lock %s' % LOCK)
def pusti_lock():
    try:
        if open(LOCK).read().strip() == str(os.getpid()): os.remove(LOCK)
    except Exception: pass
def stanje():
    try: return json.load(open(O + '/stanje.json'))
    except Exception: return {}
def spremi(st): json.dump(st, open(O + '/stanje.json', 'w'), indent=1)
def meta(p):
    import cv2
    c = cv2.VideoCapture(p)
    f = float(c.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    return int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)), f, int(c.get(cv2.CAP_PROP_FRAME_COUNT))
MUXRX = {'width': re.compile(r"""(set_property\(\s*['"]width['"]\s*,\s*)1920(\s*\))"""),
         'height': re.compile(r"""(set_property\(\s*['"]height['"]\s*,\s*)1080(\s*\))""")}
def native_copy(src, dst, w, h):
    t = open(src).read(); t2 = t
    for k, v in (('width', w), ('height', h)):
        t2, n = MUXRX[k].subn(lambda m, v=v: m.group(1) + str(v) + m.group(2), t2)
        if n != 1: sys.exit('!! in %s, not exactly one streammux setting "%s" was found (found %d) -- report this' % (os.path.basename(src), k, n))
    a = t.splitlines(); b = t2.splitlines(); diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(a) != len(b) or len(diff) > 2: sys.exit('!! copy %s differs in more than 2 lines -- report this' % os.path.basename(src))
    open(dst, 'w').write(t2)
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
        if x is not None and not okf(x): bad.append('%s = %g (should be switched off)' % (n, x))
    for n in FINAL:
        if getv(txt, n)[1] is None: bad.append('%s (final variant) not found exactly once' % n)
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
        b = p + '.p60bak'
        if not os.path.exists(b): continue
        if os.path.exists(p) and sha(p) == sha(b): os.remove(b)
        else:
            if os.path.exists(p): os.replace(p, p + '.p60_nepoznat')
            os.replace(b, p); print('   %s: RESTORED from the backup copy' % os.path.basename(p), flush=True)

# ======================================================================= FVessel sequences
def snimke():
    out = []
    for d in sorted(glob.glob(os.path.join(FV, 'clip-*'))):
        v = [x for x in glob.glob(d + '/*.mp4')]
        g = glob.glob(d + '/gt/*_gt_tracking.txt')
        if len(v) == 1 and len(g) == 1: out.append((os.path.basename(d), v[0], g[0]))
        else: print('warning: %s skipped (videos %d, gt_tracking %d)' % (d, len(v), len(g)), flush=True)
    if not out: sys.exit('!! no clip-XX sequences with a video and gt/*_gt_tracking.txt in %s' % FV)
    return out
def outp(k, ime): return O + '/log_p60_%s_%s.json' % (k, ime)

# ======================================================================= GPU
def runs():
    os.makedirs(O, exist_ok=True); uzmi_lock()
    for lk in LOCKS:
        if alive(lk): sys.exit('!! %s is still running -- run p60 when it finishes.' % os.path.basename(lk))
    try:
        used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
        print('GPU memory in use: %d MiB' % used, flush=True)
        if used > 1500: sys.exit('GPU is busy (>1500 MiB) -- something else is running on the GPU. Wait until it finishes.')
    except (IndexError, ValueError, FileNotFoundError): print('warning: nvidia-smi gave no data, continuing', flush=True)
    if not os.path.isdir('/usr/include/eigen3'):
        sys.exit('!! no Eigen (needed to compile our tracker). First run:  apt-get install -y libeigen3-dev')
    if os.path.exists(BAK) or any(os.path.exists(p + '.p60bak') for p in PROT0):
        print('the previous run was interrupted -- restoring track.h and checking the engines:', flush=True); vrati()
    SN = snimke()
    need = [PM_SRC, PT_SRC, CFG_OWN, INFER, ENG11, SRC] + [c for _, c, _ in NV] + ([ENGC] if ENGC else [])
    miss = [p for p in need if not os.path.exists(p)]
    if miss: sys.exit('!! missing:\n   ' + '\n   '.join(miss))
    res = {meta(v)[:2] for _, v, _ in SN}
    if len(res) != 1: sys.exit('!! the sequences do not have the same resolution (%s)' % sorted(res))
    w, h = res.pop(); print('sequences: %d, resolution %dx%d' % (len(SN), w, h), flush=True)
    orig = open(SRC).read(); bad = check_base(orig)
    if bad: sys.exit('!! track.h is NOT at the baseline configuration -- nothing was run:\n   ' + '\n   '.join(bad))
    for src, dst in ((PM_SRC, PM), (PT_SRC, PTJ)): native_copy(src, dst, w, h)
    st = stanje(); st.setdefault('vrijeme', {}); spremi(st)
    todo = [('nv_' + k, PTJ, c, lab) for k, c, lab in NV] + [('final', PM, CFG_OWN, NAZIV['final'])]
    todo = [(k, pr, c, lab, ime, v) for (k, pr, c, lab) in todo for (ime, v, _) in SN if not json_ok(outp(k, ime))]
    nfr = sum(meta(v)[3] for _, v, _ in SN)
    print('runs remaining: %d (in total %d frames per tracker)' % (len(todo), nfr), flush=True)
    PROT = [p for p in PROT0 if os.path.exists(p)]; sh = {}
    shutil.copy2(SRC, BAK); t0 = time.time(); preveden = False
    try:
        for p in PROT:
            sh[p] = sha(p); shutil.copy2(p, p + '.p60bak')
            if sha(p + '.p60bak') != sh[p]: sys.exit('!! backup copy %s is not valid -- STOP' % p)
        for i, (k, probe, cfg, lab, ime, v) in enumerate(todo, 1):
            if k == 'final' and not preveden:
                txt = orig
                for kk, x in FINAL.items(): txt = setv(txt, kk, x)
                open(SRC, 'w').write(txt); build(); preveden = True
                print('our tracker compiled with the final variant', flush=True)
            _, _, f, n = meta(v); out = outp(k, ime); st_ = {p: stat(p) for p in PROT}
            env = dict(os.environ, MYTRACKER_FPS='%g' % f); ta = time.time()
            try:
                r = subprocess.run(['python3', probe, v, cfg, out, INFER], cwd=W, capture_output=True, text=True, env=env,
                                   timeout=max(900, int(3 * n / f)))
                err = (r.stdout or '')[-2500:] + '\n----\n' + (r.stderr or '')[-3500:]
            except subprocess.TimeoutExpired: err = 'time limit exceeded'
            ch = [os.path.basename(p) for p in PROT if stat(p) != st_[p]]
            if ch: open(O + '/greska.txt', 'w').write(err); sys.exit('!! %s changed during the run -- report this' % ', '.join(ch))
            if not json_ok(out):
                open(O + '/greska.txt', 'w').write('%s %s\n' % (lab, ime) + err)
                sys.exit('!! %s on %s failed -- see: tail -40 %s/greska.txt' % (lab, ime, O))
            dt = time.time() - ta; st['vrijeme']['%s|%s' % (k, ime)] = [n, dt]; spremi(st)
            el = time.time() - t0
            print('  [%2d/%d] %-18s %s ok  (%.0f FPS; about %.0f min left)' % (i, len(todo), lab, ime, n / max(dt, 1e-6), el / i * (len(todo) - i) / 60), flush=True)
    finally:
        if os.path.exists(BAK): shutil.copy(BAK, SRC)
        okb = True
        if preveden:                                  # compiled only if track.h was changed
            try: build()
            except Exception as e: okb = False; print('!! compilation of the baseline track.h failed: %s' % e, flush=True)
        if okb and os.path.exists(BAK) and open(SRC).read() == open(BAK).read(): os.remove(BAK)
        print('track.h restored to the baseline configuration%s.' % ((' and compiled' if preveden else ' (it was not changed at all)') if okb else ' (COMPILATION FAILED -- report this)'), flush=True)
        for p, hs in sh.items():
            b = p + '.p60bak'
            if os.path.exists(p) and sha(p) == hs:
                if os.path.exists(b): os.remove(b)
            elif os.path.exists(b):
                if os.path.exists(p): os.replace(p, p + '.p60_nepoznat')
                os.replace(b, p); print('   !! %s changed -- RESTORED from the backup copy; report this' % os.path.basename(p), flush=True)
        pusti_lock()
    print('GPU part done (%.0f min).' % ((time.time() - t0) / 60), flush=True)

# ======================================================================= evaluation
def ucitaj(p):
    d = json.load(open(p))
    if isinstance(d, dict): d = d.get('frames') or d.get('data') or list(d.values())[0]
    out = {}
    for fd in d:
        dets = fd.get('detections', fd.get('objects', []))
        if dets:
            out[int(fd['frame'])] = [(int(x['id']), [float(q) for q in (x.get('box') or x.get('bbox'))[:4]],
                                      None if x.get('conf', x.get('score')) is None else float(x.get('conf', x.get('score')))) for x in dets]
    return out
def ucitaj_gt(p):
    g = {}
    for l in open(p):
        s = l.strip().split(',')
        if len(s) < 6: continue
        g.setdefault(int(float(s[0])), []).append((int(float(s[1])), [float(s[2]), float(s[3]), float(s[4]), float(s[5])]))
    return g
def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[0] + a[2], b[0] + b[2]); y2 = min(a[1] + a[3], b[1] + b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1); u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0
def pravilo(tr, fps, Wk, Hk, c_big=None):
    """same as p58 (causal); returns the set of (frame, id) that is not reported"""
    p = PRAVILO; po = {}
    for f, L_ in tr.items():
        for (t, b, c) in L_: po.setdefault(t, []).append((f, b, c))
    pot = set()
    for t, rows in po.items():
        rows.sort(key=lambda r: r[0]); f0 = rows[0][0]; cx = []; cy = []; ww = []; hh = []; cc = []
        for (f, b, c) in rows:
            cx.append(b[0] + b[2] / 2); cy.append(b[1] + b[3] / 2); ww.append(b[2]); hh.append(b[3])
            if c is not None: cc.append(c)
            if (f - f0) / fps < p['T_MIN']: continue
            vel = np.sqrt(np.median(ww) * np.median(hh))
            if vel <= 0: continue
            ax = np.array(cx); ay = np.array(cy)
            if np.percentile(np.hypot(ax - np.median(ax), ay - np.median(ay)), 90) >= p['STATIC'] * vel: continue
            mc = np.median(cc) if cc else None
            nisko = mc is not None and mc < p['C_LOW']
            golemo = np.median(np.array(ww) * np.array(hh)) > p['A_BIG'] * Wk * Hk and (c_big is None or (mc is not None and mc < c_big))
            if nisko or golemo: pot.add((f, t))
    return pot
def gated(tr, pot): return {f: [r for r in L_ if (f, r[0]) not in pot] for f, L_ in tr.items()}

def te(items):
    try:
        import trackeval
    except ImportError:
        sys.path.insert(0, W + '/TrackEval'); import trackeval
    TMP = O + '/te_tmp'
    if os.path.exists(TMP): shutil.rmtree(TMP)
    si = {}
    for seq, (G, T) in items.items():
        si[seq] = max(list(G) + list(T) + [1])
        os.makedirs('%s/gt/%s/gt' % (TMP, seq)); os.makedirs('%s/tr/x/data' % TMP, exist_ok=True)
        with open('%s/gt/%s/gt/gt.txt' % (TMP, seq), 'w') as f:
            for t in sorted(G):
                for i, b in G[t]: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,1,1\n' % (t, i, *b))
        with open('%s/tr/x/data/%s.txt' % (TMP, seq), 'w') as f:
            for t in sorted(T):
                for i, b in T[t]: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n' % (t, i, *b))
    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update(PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False,
              TIME_PROGRESS=False, USE_PARALLEL=False, DISPLAY_LESS_PROGRESS=True)
    MET = [trackeval.metrics.HOTA({'PRINT_CONFIG': False}), trackeval.metrics.CLEAR({'PRINT_CONFIG': False}), trackeval.metrics.Identity({'PRINT_CONFIG': False})]
    dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dc.update(GT_FOLDER=TMP + '/gt', TRACKERS_FOLDER=TMP + '/tr', BENCHMARK='b', SPLIT_TO_EVAL='', SEQ_INFO=si, TRACKERS_TO_EVAL=['x'],
              DO_PREPROC=False, SKIP_SPLIT_FOL=True, GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt', TRACKER_SUB_FOLDER='data', PRINT_CONFIG=False)
    out, _ = trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)], MET)
    R = out['MotChallenge2DBox']['x']; res = {}
    for s in si:
        r = R[s]['pedestrian']
        res[s] = dict(HOTA=float(np.mean(r['HOTA']['HOTA'])), DetA=float(np.mean(r['HOTA']['DetA'])), AssA=float(np.mean(r['HOTA']['AssA'])),
                      IDF1=float(r['Identity']['IDF1']), IDSW=float(r['CLEAR']['IDSW']), FP=float(r['CLEAR']['CLR_FP']),
                      FN=float(r['CLEAR']['CLR_FN']), TP=float(r['CLEAR']['CLR_TP']))
    shutil.rmtree(TMP, ignore_errors=True)
    return res

def uzorak(G, T, fps, k):
    """GT per second -> 1 Hz sequence; tracker output from frame round(s*fps)+k. Returns (GT, T) with time t = s - smin + 1."""
    s0, s1 = min(G), max(G); Gs = {}; Ts = {}
    for s in range(s0, s1 + 1):
        t = s - s0 + 1
        if s in G: Gs[t] = G[s]
        f = int(round(s * fps)) + k
        if f in T: Ts[t] = [(i, b) for (i, b, c) in T[f]]
    return Gs, Ts
def poklapanje(G, T, fps, k):
    """sum of the best IoU for each GT box (continuous, so the peak is clear also for slow vessels)"""
    n = 0.0
    for s, gl in G.items():
        f = int(round(s * fps)) + k
        tl = T.get(f, [])
        for _, gb in gl:
            if tl: n += max(iou(gb, tb) for (_, tb, _) in tl)
    return n

def analiza():
    from scipy.stats import wilcoxon
    SN = snimke(); st = stanje()
    P('p60 -- FVessel (%s), %d sequences; tracker values frozen (IEEE final variant), nothing is tuned' % (FV, len(SN)))
    P('Detector: %s; config %s' % (DET_OPIS, INFER))
    LOG = {}; GT = {}; MT = {}
    kljucevi = ['final'] + ['nv_' + k for k, _, _ in NV]
    for ime, v, g in SN:
        GT[ime] = ucitaj_gt(g); MT[ime] = meta(v)
        for k in kljucevi:
            p = outp(k, ime)
            if json_ok(p): LOG[(k, ime)] = ucitaj(p)
    imaju = [k for k in kljucevi if all((k, ime) in LOG for ime, _, _ in SN)]
    if 'final' not in imaju: sys.exit('!! no logs of our tracker for all sequences -- first run without --analiza')
    for k in kljucevi:
        if k not in imaju: P('warning: %s does not have logs for all sequences -- left out of the comparison' % NAZIV[k])
    # alignment second -> frame: one k for the whole set, by the summed match of all trackers
    K = range(-50, 51); zb = {k: 0 for k in K}; po = {}
    for ime, v, g in SN:
        fps = MT[ime][2]
        for k in K:
            n = sum(poklapanje(GT[ime], LOG[(t, ime)], fps, k) for t in imaju); zb[k] += n
            po.setdefault(ime, {})[k] = n
    kbest = max(K, key=lambda k: (round(zb[k], 6), -abs(k)))
    P('Alignment: frame = round(second x fps) + k; best k for the whole set = %+d (sum of IoU %.1f; k-5: %.1f, k+5: %.1f; k = 0: %.1f)' % (
        kbest, zb[kbest], zb.get(kbest - 5, 0), zb.get(kbest + 5, 0), zb[0]))
    P('  best k per sequence: ' + ', '.join('%s %+d' % (ime, max(K, key=lambda k: (round(po[ime][k], 6), -abs(k)))) for ime, _, _ in SN))
    if POMAK is not None:
        P('*** SENSITIVITY: the GIVEN offset k = %+d is used (sum of IoU %.1f), not the best k = %+d; the main result is not touched ***' % (
            POMAK, zb.get(POMAK, float('nan')), kbest))
        kbest = POMAK
    # rule p58 on the output of ours
    for ime, v, g in SN:
        Wk, Hk, fps, n = MT[ime]; T = LOG[('final', ime)]
        LOG[('final_v1', ime)] = gated(T, pravilo(T, fps, Wk, Hk)); LOG[('final_v2', ime)] = gated(T, pravilo(T, fps, Wk, Hk, c_big=0.7))
    sustavi = imaju[:1] + ['final_v1', 'final_v2'] + imaju[1:]
    REZ = {}
    for k in sustavi:
        items = {}
        for ime, v, g in SN:
            items[ime] = uzorak(GT[ime], LOG[(k, ime)], MT[ime][2], kbest)
        REZ[k] = te(items)
    json.dump(dict(k=kbest, rez=REZ), open(O + ('/rezultati.json' if POMAK is None else '/rezultati_pomak%+d.json' % POMAK), 'w'))
    # description of the set
    ng = sum(len(b) for ime in GT for b in GT[ime].values()); nid = sum(len({i for b in GT[ime].values() for i, _ in b}) for ime in GT)
    hh = [b[3] for ime in GT for L_ in GT[ime].values() for _, b in L_]
    P('Set: %d sequences, %.1f min of video, %d GT boxes (1 Hz), %d tracks; median box height %.0f px (%dx%d)' % (
        len(SN), sum(MT[i][3] / MT[i][2] for i, _, _ in SN) / 60, ng, nid, np.median(hh), MT[SN[0][0]][0], MT[SN[0][0]][1]))
    MET = ['HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW', 'FP', 'FN']
    def mean(k, m): return float(np.mean([REZ[k][s][m] for s in REZ[k]]))
    P('\n=== MEAN PER SEQUENCE (n = %d; IDSW, FP, FN = sum)' % len(SN))
    P('  %-24s %6s %6s %6s %6s %6s %6s %6s %6s' % ('system', 'HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW', 'FP', 'FN', 'FPS'))
    for k in sustavi:
        fpsv = [st.get('vrijeme', {}).get('%s|%s' % (k, ime)) for ime, _, _ in SN]
        fpsv = [x for x in fpsv if x]; fr = (sum(x[0] for x in fpsv) / sum(x[1] for x in fpsv)) if fpsv else float('nan')
        P('  %-24s %6.3f %6.3f %6.3f %6.3f %6.0f %6.0f %6.0f %6s' % (NAZIV[k], mean(k, 'HOTA'), mean(k, 'DetA'), mean(k, 'AssA'), mean(k, 'IDF1'),
            sum(REZ[k][s]['IDSW'] for s in REZ[k]), sum(REZ[k][s]['FP'] for s in REZ[k]), sum(REZ[k][s]['FN'] for s in REZ[k]),
            '%.0f' % fr if fr == fr else '-'))
    P('\n=== OURS FINAL minus NVIDIA (over sequences, two-sided Wilcoxon, Holm over %d comparisons per measure; n = %d)' % (len(imaju) - 1, len(SN)))
    for m in ('HOTA', 'IDF1', 'AssA', 'DetA', 'IDSW'):
        ps = []; rows = []
        for k in imaju[1:]:
            d = np.array([REZ['final'][s][m] - REZ[k][s][m] for s in sorted(REZ['final'])])
            pv = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            ps.append(pv); rows.append((k, d))
        o = np.argsort(ps); hp = [0] * len(ps); run = 0
        for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); hp[i] = run
        for (k, d), pv, h in zip(rows, ps, hp):
            bolje = int((d > 0).sum()) if m != 'IDSW' else int((d < 0).sum())
            P('  %-5s vs %-12s difference %+.3f  better on %d/%d  p = %.3f  Holm %.3f' % (m, NAZIV[k], d.mean(), bolje, len(d), pv, h))
    P('\n=== TRACK CONFIRMATION RULE (p58) ON FVESSEL')
    for k in ('final_v1', 'final_v2'):
        dfp = sum(REZ[k][s]['FP'] - REZ['final'][s]['FP'] for s in REZ[k]); dtp = sum(REZ[k][s]['TP'] - REZ['final'][s]['TP'] for s in REZ[k])
        P('  %-24s FP %+5.0f  TP %+5.0f  HOTA %+.4f  (TP drop = real vessels lost on the sampled frames)' % (
            NAZIV[k], dfp, dtp, mean(k, 'HOTA') - mean('final', 'HOTA')))
    P('\nPer sequence (HOTA): ' + ' | '.join('%s ' % ime + ' '.join('%.3f' % REZ[k][ime]['HOTA'] for k in sustavi) for ime, _, _ in SN))
    P('(order in the row: ' + ', '.join(NAZIV[k] for k in sustavi) + ')')
    P('Note: the FVessel GT is 1 Hz, so IDSW and FP are measured only on the frames of each second; detector: %s.' % DET_OPIS)
    open(REZTXT, 'w').write('\n'.join(L) + '\n')
    print('\nresult: %s' % REZTXT)

if __name__ == '__main__':
    if '--vrati' in sys.argv: vrati(); sys.exit(0)
    if '--analiza' not in sys.argv and POMAK is None: runs()
    analiza()
