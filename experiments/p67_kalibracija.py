#!/usr/bin/env python3
"""p67 -- MIPRO 2027: CONFIDENCE AND CALIBRATION OF THE YOLO11m DETECTOR AND CLASS STABILITY ON THREE MARITIME SETS.

Plan written BEFORE the run. ONLY the detector is measured (without a tracker).
Detectors (DeepStream nvinfer, TensorRT FP16, input 640, threshold 0.25 -- the same input the tracker gets in the whole paper):
  ssmot = YOLO11m trained on SSMOT (config_infer_11m.txt)
  coco  = general COCO YOLO11m, class boat only (config_infer_coco11m_640.txt)
Sets (streammux at the native resolution): SSMOT (16 test sequences), SMD-Plus VIS_Onshore (40), FVessel Clip-10 (10, GT 1 Hz).

Run (GPU free; estimate 1-1.5 h: 132 DeepStream runs, ~192 000 frames):
    nohup python3 /workspace/p67_kalibracija.py > /workspace/p67_out.txt 2>&1 &
Monitoring:           tail -3 /workspace/p67_out.txt
Analysis only (CPU):  python3 /workspace/p67_kalibracija.py --analiza
FVessel, same offset for both detectors (p67b, CPU, ~5 min):  python3 /workspace/p67_kalibracija.py --fvessel-pomaci
Calibration against IoU (p67c, CPU, ~5 min):  python3 /workspace/p67_kalibracija.py --laece
If interrupted:       python3 /workspace/p67_kalibracija.py --vrati      (then rerun -- it resumes)
Output: /workspace/scpilot/p67_kalibracija.txt  and  /workspace/scpilot/p67/p67_rezultati.json (data for the figures)
"""
import os, re, sys, json, glob, time, shutil, hashlib, subprocess, atexit, collections, warnings
warnings.filterwarnings('ignore')
import numpy as np

W = os.environ.get('P67_W', '/workspace'); S = W + '/scpilot'; O = S + '/p67'
DD = O + '/det'; RAD = O + '/rad'; LOCK = O + '/p67.lock'; PROBE = O + '/probe_det.py'
OUT_TXT = S + '/p67_kalibracija.txt'; OUT_JSON = O + '/p67_rezultati.json'
ENG11 = W + '/det_glavni/B_yolo11m/weights/best_11m_b1_gpu0_fp16.engine'
ENGC = W + '/coco_det/yolo11m_coco_640_b1_gpu0_fp16.engine'
DETS = {'ssmot': (W + '/DeepStream-Yolo/config_infer_11m.txt', ENG11, 'YOLO11m trained on SSMOT (9 classes)'),
        'coco': (W + '/DeepStream-Yolo/config_infer_coco11m_640.txt', ENGC, 'general COCO YOLO11m, class boat only')}
PROT0 = [W + '/model_b1_gpu0_fp16.engine', W + '/best_0904.engine', W + '/osnet_vessel.engine', ENG11, ENGC]
LOCKS = [S + '/p54/p54.lock', S + '/p55/p55.lock', S + '/p56/p56.lock', S + '/p59/p59.lock', S + '/p60/p60.lock',
         S + '/p60_coco/p60.lock', S + '/p64_coco/p60.lock', S + '/p64_ssmot/p60.lock', S + '/p65/p65.lock', S + '/p66/p66.lock']
SS16 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
SS_VID = W + '/SSMOT_video/video/SSMOT_%d.mp4'; SS_GT = W + '/SSMOT_video/ground-truth/SSMOT_%d/gt/gt.txt'
SMD = W + '/SMD_mot'
FVD = W + '/Clip-10_h264' if os.path.isdir(W + '/Clip-10_h264') else W + '/Clip-10'
P54 = S + '/p54/det/ds_mux_%d.json'
PLOVILA = {'ferry', 'vessel/ship', 'boat', 'kayak', 'sail boat', 'sailboat', 'speed boat'}
DVOSMISLENE = ['MVI_1584_VIS', 'MVI_1615_VIS']
SKUPOVI = ['ssmot', 'smd', 'fvessel']
NAZ_SK = {'ssmot': 'SSMOT', 'smd': 'SMD-Plus', 'fvessel': 'FVessel'}
THR = 0.25; TAU = 0.8; NB = 15; NBOOT = 2000; SEED = 67
RAZ_KL = [(0.25, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.0001)]
T_MIN_TRAG = 10
VEL = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 1.01)]   # box height / image height

PROBE_SRC = r'''import sys, json, gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds
Gst.init(None)
VIDEO, OUT, INFER, MW, MH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
log = []; n = [0]; uid = [0]
def probe_fn(pad, info, u):
    buf = info.get_buffer()
    if not buf: return Gst.PadProbeReturn.OK
    bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not bm: return Gst.PadProbeReturn.OK
    lf = bm.frame_meta_list
    while lf:
        try: fm = pyds.NvDsFrameMeta.cast(lf.data)
        except StopIteration: break
        n[0] += 1; dets = []; lo = fm.obj_meta_list
        while lo:
            try: o = pyds.NvDsObjectMeta.cast(lo.data)
            except StopIteration: break
            uid[0] += 1; r = o.rect_params
            dets.append({'id': uid[0], 'class': int(o.class_id), 'box': [float(r.left), float(r.top), float(r.width), float(r.height)], 'conf': float(o.confidence)})
            try: lo = lo.next
            except StopIteration: break
        log.append({'frame': n[0], 'detections': dets})
        try: lf = lf.next
        except StopIteration: break
    return Gst.PadProbeReturn.OK
pl = Gst.Pipeline()
src = Gst.ElementFactory.make("filesrc", "source"); dec = Gst.ElementFactory.make("decodebin", "decoder")
mux = Gst.ElementFactory.make("nvstreammux", "mux"); pgie = Gst.ElementFactory.make("nvinfer", "pgie"); sink = Gst.ElementFactory.make("fakesink", "sink")
src.set_property('location', VIDEO)
mux.set_property('width', MW); mux.set_property('height', MH); mux.set_property('batch-size', 1); mux.set_property('batched-push-timeout', 4000000)
pgie.set_property('config-file-path', INFER); sink.set_property('sync', False)
for e in (src, dec, mux, pgie, sink): pl.add(e)
src.link(dec)
def on_pad(d, pad, m):
    caps = pad.get_current_caps() or pad.query_caps(None)
    if caps and not caps.get_structure(0).get_name().startswith('video'): return
    sp = m.get_request_pad("sink_0")
    if not sp.is_linked(): pad.link(sp)
dec.connect("pad-added", on_pad, mux)
mux.link(pgie); pgie.link(sink)
pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe_fn, 0)
loop = GLib.MainLoop(); ok = [False]
def bus_call(bus, msg, lp):
    if msg.type == Gst.MessageType.EOS:
        json.dump(log, open(OUT, 'w')); ok[0] = True; print("frames: %d, detections: %d" % (n[0], uid[0])); lp.quit()
    elif msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error(); print("ERROR: %s | %s" % (err, dbg)); lp.quit()
    return True
bus = pl.get_bus(); bus.add_signal_watch(); bus.connect("message", bus_call, loop)
pl.set_state(Gst.State.PLAYING)
try: loop.run()
except Exception: pass
pl.set_state(Gst.State.NULL)
sys.exit(0 if ok[0] else 1)
'''

# ============================================================================ helpers
L = []
def P(x=''):
    print(x, flush=True); L.append(str(x))
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
    try: pid = open(lock).read().strip()
    except Exception: return None
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
                sys.exit('!! p67 is already running (PID %s) -- do NOT run twice.' % (pid or '?'))
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
_META = {}
def meta(p):
    if p not in _META:
        import cv2
        c = cv2.VideoCapture(p)
        f = float(c.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        _META[p] = (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)), f, int(c.get(cv2.CAP_PROP_FRAME_COUNT)))
    return _META[p]
def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[0] + a[2], b[0] + b[2]); y2 = min(a[1] + a[3], b[1] + b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1); u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0
def vrati():
    for p in PROT0:
        b = p + '.p67bak'
        if not os.path.exists(b): continue
        if os.path.exists(p) and sha(p) == sha(b): os.remove(b); print('   %s: untouched, copy deleted' % os.path.basename(p), flush=True)
        else:
            if os.path.exists(p): os.replace(p, p + '.p67_nepoznat')
            os.replace(b, p); print('   %s: RESTORED from the backup copy' % os.path.basename(p), flush=True)

# ============================================================================ sequences
def snimke(sk):
    """list (name, video, gt)"""
    out = []
    if sk == 'ssmot':
        for s in SS16: out.append(('SSMOT_%d' % s, SS_VID % s, SS_GT % s))
    elif sk == 'smd':
        for d in sorted(glob.glob(SMD + '/MVI_*')):
            ime = os.path.basename(d); v = '%s/%s.mp4' % (d, ime); g = d + '/gt/gt.txt'
            if os.path.exists(v) and os.path.exists(g): out.append((ime, v, g))
    else:
        for d in sorted(glob.glob(FVD + '/clip-*')):
            v = glob.glob(d + '/*.mp4'); g = glob.glob(d + '/gt/*_gt_tracking.txt')
            if len(v) == 1 and len(g) == 1: out.append((os.path.basename(d), v[0], g[0]))
    return out
def outp(det, sk, ime): return DD + '/%s_%s_%s.json' % (det, sk, ime)

# ============================================================================ GPU
def runs():
    os.makedirs(DD, exist_ok=True); os.makedirs(RAD, exist_ok=True); uzmi_lock()
    for lk in LOCKS:
        if alive(lk): sys.exit('!! %s is still running -- run p67 when it finishes.' % lk)
    if os.environ.get('P67_NOGPU') != '1':
        try:
            used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
            print('GPU memory in use: %d MiB' % used, flush=True)
            if used > 1500: sys.exit('GPU is busy (>1500 MiB) -- something else is running on the GPU. Wait until it finishes.')
        except (IndexError, ValueError, FileNotFoundError): print('warning: nvidia-smi gave no data, continuing', flush=True)
    if any(os.path.exists(p + '.p67bak') for p in PROT0):
        print('the previous run was interrupted -- checking the engines:', flush=True); vrati()
    miss = [DETS[d][0] for d in DETS if not os.path.exists(DETS[d][0])] + [DETS[d][1] for d in DETS if not os.path.exists(DETS[d][1])]
    if miss: sys.exit('!! missing:\n   ' + '\n   '.join(miss))
    SN = {sk: snimke(sk) for sk in SKUPOVI}
    for sk in SKUPOVI:
        miss = [p for _, v, g in SN[sk] for p in (v, g) if not os.path.exists(p)]
        if miss or not SN[sk]: sys.exit('!! set %s: no sequences or missing:\n   %s' % (NAZ_SK[sk], '\n   '.join(miss[:10])))
        res = {meta(v)[:2] for _, v, _ in SN[sk]}
        if len(res) != 1: sys.exit('!! %s: the sequences do not have the same resolution (%s) -- report this' % (NAZ_SK[sk], sorted(res)))
        print('%-9s %2d sequences, %dx%d, %d frames' % (NAZ_SK[sk], len(SN[sk]), *list(res)[0], sum(meta(v)[3] for _, v, _ in SN[sk])), flush=True)
    if shutil.disk_usage(W).free / 2 ** 30 < 3: sys.exit('!! less than 3 GB free on the disk')
    st = stanje(); st.setdefault('vrijeme', {}); st.setdefault('cfg_sha', {})
    for d in DETS:
        h = sha(DETS[d][0])
        if d in st['cfg_sha'] and st['cfg_sha'][d] != h: sys.exit('!! %s changed since the first p67 run -- report this' % DETS[d][0])
        st['cfg_sha'][d] = h
    spremi(st)
    if os.environ.get('P67_PROBE_MOCK'): shutil.copy(os.environ['P67_PROBE_MOCK'], PROBE)
    else: open(PROBE, 'w').write(PROBE_SRC)
    todo = [(d, sk, ime, v) for d in DETS for sk in SKUPOVI for ime, v, _ in SN[sk] if not json_ok(outp(d, sk, ime))]
    print('runs remaining: %d (2 detectors x 3 sets)' % len(todo), flush=True)
    PROT = [p for p in PROT0 if os.path.exists(p)]; sh = {}; t0 = time.time(); nfr = [0]
    tot = sum(meta(v)[3] for _, _, _, v in todo) or 1
    try:
        for p in PROT:
            sh[p] = sha(p); shutil.copy2(p, p + '.p67bak')
            if sha(p + '.p67bak') != sh[p]: sys.exit('!! backup copy %s is not valid -- STOP' % p)
        for i, (d, sk, ime, v) in enumerate(todo, 1):
            mw, mh, f, n = meta(v); out = outp(d, sk, ime); ok = False
            for att in (1, 2):
                st_ = {p: stat(p) for p in PROT}; ta = time.time()
                try:
                    r = subprocess.run(['python3', PROBE, v, out + '.tmp', DETS[d][0], str(mw), str(mh)], cwd=RAD, capture_output=True,
                                       text=True, timeout=max(900, int(3 * n / f)))
                    err = (r.stdout or '')[-2500:] + '\n----\n' + (r.stderr or '')[-3500:]
                except subprocess.TimeoutExpired: err = 'time limit exceeded'
                ch = [os.path.basename(p) for p in PROT if stat(p) != st_[p]]
                if ch:
                    if os.path.exists(out + '.tmp'): os.remove(out + '.tmp')
                    open(O + '/greska.txt', 'w').write(err)
                    sys.exit('!! %s changed during the run (%s, %s) -- restoring from the copy; see: tail -40 %s/greska.txt' % (', '.join(ch), d, ime, O))
                if json_ok(out + '.tmp'):
                    os.replace(out + '.tmp', out); st['vrijeme']['%s|%s|%s' % (d, sk, ime)] = [n, time.time() - ta]; spremi(st); ok = True; break
                if os.path.exists(out + '.tmp'): os.remove(out + '.tmp')
                open(O + '/greska.txt', 'w').write('%s %s %s\n' % (d, sk, ime) + err)
            if not ok: sys.exit('!! %s / %s failed in the second attempt too -- see: tail -40 %s/greska.txt' % (d, ime, O))
            nfr[0] += n; el = time.time() - t0
            print('  [%3d/%d] %-5s %-9s %-14s ok (%.0f FPS; about %.0f min left)' % (i, len(todo), d, NAZ_SK[sk], ime, n / max(time.time() - ta, 1e-6),
                  el / nfr[0] * (tot - nfr[0]) / 60), flush=True)
    finally:
        for p, hh in sh.items():
            b = p + '.p67bak'
            if os.path.exists(p) and sha(p) == hh:
                if os.path.exists(b): os.remove(b)
            elif os.path.exists(b):
                if os.path.exists(p): os.replace(p, p + '.p67_nepoznat')
                os.replace(b, p); print('   !! %s changed -- RESTORED from the backup copy; report this' % os.path.basename(p), flush=True)
        pusti_lock()
    print('GPU part done (%.0f min).' % ((time.time() - t0) / 60), flush=True)

# ============================================================================ loading
def ucitaj_det(p):
    out = {}
    for fd in json.load(open(p)):
        dets = fd.get('detections', [])
        if dets: out[int(fd['frame'])] = [(int(x['class']), [float(q) for q in x['box'][:4]], float(x['conf'])) for x in dets]
    return out
def gt_ssmot(p, det):
    def load(s):
        g = collections.defaultdict(list)
        for l in open(p):
            f = l.strip().split(',')
            if len(f) < 6: continue
            g[int(float(f[0]))].append((int(float(f[1])), [float(q) * s for q in f[2:6]]))
        return g
    best = (1.0, -1)
    for s in (1.0, 1.5):
        g = load(s); fr = sorted(set(g) & set(det))[:400]
        h = sum(1 for t in fr for _, b in g[t] if any(iou(b, x[1]) > 0.5 for x in det[t]))
        if h > best[1]: best = (s, h)
    return load(best[0]), {}, best[0]
_KL = None
def gt_smd(p):
    global _KL
    if _KL is None: _KL = {int(k): v for k, v in json.load(open(SMD + '/klase.json')).items()}
    Gv = collections.defaultdict(list); Gd = collections.defaultdict(list)
    for l in open(p):
        s = l.strip().split(',')
        if len(s) < 8: continue
        t, k, c = int(s[0]), int(s[1]), int(s[7]); b = [float(q) for q in s[2:6]]
        (Gv if _KL.get(c, '').lower() in PLOVILA else Gd)[t].append((k, b))
    return Gv, Gd
def gt_fv(p):
    g = collections.defaultdict(list)
    for l in open(p):
        s = l.strip().split(',')
        if len(s) < 6: continue
        g[int(float(s[0]))].append((int(float(s[1])), [float(q) for q in s[2:6]]))
    return g
def nazivi_klasa(cfg):
    try:
        txt = open(cfg).read(); m = re.search(r'^\s*labelfile-path\s*=\s*(\S+)', txt, re.M)
        if not m: return {}
        p = m.group(1); p = p if os.path.isabs(p) else os.path.join(os.path.dirname(cfg), p)
        return {i: l.strip() for i, l in enumerate(open(p)) if l.strip()}
    except Exception: return {}

# ============================================================================ matching
def upari(dets, gv, gd, thr):
    """greedy by decreasing confidence; returns a list (class, conf, y, gt_id) for the detections that are not left out, and the number of GT vessels.
    y = 1 (TP, matched to a vessel), 0 (FP); those matched to a distractor are left out."""
    G = [(gid, b, True) for gid, b in gv] + [(None, b, False) for b in gd]
    used = [False] * len(G); res = []
    for (c, b, p) in sorted(dets, key=lambda x: -x[2]):
        bi, bj = thr, -1
        for j, (gid, gb, pl) in enumerate(G):
            if used[j]: continue
            v = iou(b, gb)
            if v >= bi: bi, bj = v, j
        if bj < 0: res.append((c, p, 0, None, b[3]))
        else:
            used[bj] = True
            if G[bj][2]: res.append((c, p, 1, G[bj][0], b[3]))
    return res

class Kombinacija:
    """everything measured for one detector x set; per sequence for the cluster bootstrap"""
    def __init__(self): self.v = {}
    def snimka(self, ime): return self.v.setdefault(ime, dict(conf=[], y=[], hrel=[], ngt=0, par=[], tr_kl=collections.defaultdict(list),
                                                              tr_cf=collections.defaultdict(list), lanci=[]))

def pomak(Dt, gtf, opseg, frejm_od_gt):
    """best offset: sum of the best IoU (GT box -> detection); preference for the smaller |offset|"""
    zb = {}
    for o in opseg:
        n = 0.0
        for t, gl in gtf.items():
            dl = Dt.get(frejm_od_gt(t) + o, [])
            if dl:
                for _, b in gl: n += max(iou(b, x[1]) for x in dl)
        zb[o] = n
    return max(zb, key=lambda x: (round(zb[x], 3), -abs(x))), zb

def lanci(Dt):
    """without GT: detections of neighbouring frames linked with the Hungarian algorithm, IoU >= 0.5 -> (class_t, class_t+1, min_conf)"""
    from scipy.optimize import linear_sum_assignment
    out = []; fr = sorted(Dt)
    for t in fr:
        a = Dt.get(t); b = Dt.get(t + 1)
        if not a or not b: continue
        M = np.array([[iou(x[1], y[1]) for y in b] for x in a]); r, c = linear_sum_assignment(-M)
        for i, j in zip(r, c):
            if M[i, j] >= 0.5: out.append((a[i][0], b[j][0], min(a[i][2], b[j][2])))
    return out

def mjeri(det, sk, SN, thr=0.5, pomak_sila=None, bez=(), s_lancima=True):
    K = Kombinacija(); info = {}
    for ime, v, g in SN:
        if ime in bez: continue
        Dt = ucitaj_det(outp(det, sk, ime)); mw, mh, fps, n = meta(v); s = K.snimka(ime)
        if sk == 'ssmot':
            Gv, Gd, sc = gt_ssmot(g, Dt); info.setdefault('mjerilo', {})[ime] = sc
            okviri = sorted(set(range(1, n + 1)) | set(Gv)); f_od = lambda t: t
        elif sk == 'smd':
            Gv, Gd = gt_smd(g); okviri = list(range(1, n + 1)); f_od = lambda t: t
        else:
            Gs = gt_fv(g); Gv = Gs; Gd = {}; okviri = sorted(Gs); f_od = lambda t, fps=fps: int(round(t * fps))
        info.setdefault('gt', {})[ime] = (Gv, Gd, f_od, okviri, Dt)
    # one offset for the whole combination
    if pomak_sila is not None: o = pomak_sila
    else:
        opseg = range(-50, 51) if sk == 'fvessel' else range(-3, 4)
        zb = collections.Counter()
        for ime, (Gv, Gd, f_od, okviri, Dt) in info['gt'].items():
            gsub = {t: Gv[t] for t in list(sorted(Gv))[::(1 if sk == 'fvessel' else 5)]}
            _, z = pomak(Dt, gsub, opseg, f_od); zb.update(z)
        o = max(zb, key=lambda x: (round(zb[x], 3), -abs(x)))
    info['pomak'] = o
    for ime, (Gv, Gd, f_od, okviri, Dt) in info['gt'].items():
        s = K.snimka(ime); prev = {}
        for t in okviri:
            dl = Dt.get(f_od(t) + o, []); gv = Gv.get(t, []); gd = [b for _, b in Gd.get(t, [])] if isinstance(Gd, dict) else []
            s['ngt'] += len(gv)
            if not dl: prev = {}; continue
            r = upari(dl, gv, gd, thr); cur = {}
            for (c, p, y, gid, hb) in r:
                s['conf'].append(p); s['y'].append(y); s['hrel'].append(hb / float(mh))
                if y == 1:
                    cur[gid] = (c, p); s['tr_kl'][gid].append(c); s['tr_cf'][gid].append(p)
                    if sk != 'fvessel' and gid in prev and prev[gid][2] == t - 1:
                        s['par'].append((prev[gid][0], c, min(prev[gid][1], p), abs(p - prev[gid][1])))
            prev = {gid: (c, p, t) for gid, (c, p) in cur.items()} if sk != 'fvessel' else {}
        if s_lancima: s['lanci'] = lanci(Dt)
    return K, info

# ============================================================================ measures
def razredi(conf, nb=NB, lo=THR):
    e = np.linspace(lo, 1.0, nb + 1); e[-1] = 1.0 + 1e-9
    return np.clip(np.digitize(conf, e) - 1, 0, nb - 1)
def hist_snimke(K, eq_mass_edges=None):
    """per sequence: per bin (count, sum of conf, sum of y); fine histogram for AUROC; counters for the threshold"""
    H = {}
    fine = np.linspace(THR, 1.0, 751); fine[-1] = 1.0 + 1e-9
    for ime, s in K.v.items():
        c = np.array(s['conf'], float); y = np.array(s['y'], int)
        if eq_mass_edges is not None: b = np.clip(np.digitize(c, eq_mass_edges) - 1, 0, len(eq_mass_edges) - 2); nb = len(eq_mass_edges) - 1
        else: b = razredi(c); nb = NB
        cnt = np.bincount(b, minlength=nb).astype(float); sc = np.bincount(b, weights=c, minlength=nb); sy = np.bincount(b, weights=y, minlength=nb)
        fb = np.clip(np.digitize(c, fine) - 1, 0, 749)
        tp = np.bincount(fb[y == 1], minlength=750).astype(float); fp = np.bincount(fb[y == 0], minlength=750).astype(float)
        H[ime] = dict(cnt=cnt, sc=sc, sy=sy, tp=tp, fp=fp, n=len(c), ntp=float(y.sum()), ngt=float(s['ngt']),
                      brier=float(((c - y) ** 2).sum()), sumc=float(c.sum()),
                      tau_n=float((c >= TAU).sum()), tau_tp=float(((c >= TAU) & (y == 1)).sum()))
    return H
def agreg(H, imena):
    keys = ['cnt', 'sc', 'sy', 'tp', 'fp']; A = {k: sum(H[i][k] for i in imena) for k in keys}
    for k in ('n', 'ntp', 'ngt', 'brier', 'sumc', 'tau_n', 'tau_tp'): A[k] = sum(H[i][k] for i in imena)
    return A
def ece(A):
    m = A['cnt'] > 0
    if A['n'] == 0: return float('nan'), float('nan')
    gap = np.abs(A['sy'][m] / A['cnt'][m] - A['sc'][m] / A['cnt'][m])
    return float((A['cnt'][m] * gap).sum() / A['n']), float(gap.max())
def auroc(A):
    tp, fp = A['tp'], A['fp']; ntp, nfp = tp.sum(), fp.sum()
    if ntp == 0 or nfp == 0: return float('nan')
    cfp = np.cumsum(fp) - fp            # FP strictly lower bin
    return float(((tp * cfp).sum() + 0.5 * (tp * fp).sum()) / (ntp * nfp))
def mjere_agr(A):
    e, mce = ece(A)
    return dict(ECE=e, MCE=mce, AUROC=auroc(A), preciznost=A['ntp'] / A['n'] if A['n'] else float('nan'),
                odziv=A['ntp'] / A['ngt'] if A['ngt'] else float('nan'), brier=A['brier'] / A['n'] if A['n'] else float('nan'),
                precijenjenost=(A['sumc'] - A['ntp']) / A['n'] if A['n'] else float('nan'),
                prec_tau=A['tau_tp'] / A['tau_n'] if A['tau_n'] else float('nan'), tp_iznad_tau=A['tau_tp'] / A['ntp'] if A['ntp'] else float('nan'),
                n=A['n'], ntp=A['ntp'], ngt=A['ngt'])
def ap50(K):
    c = np.concatenate([np.array(s['conf'], float) for s in K.v.values()] or [np.zeros(0)])
    y = np.concatenate([np.array(s['y'], int) for s in K.v.values()] or [np.zeros(0, int)]); ngt = sum(s['ngt'] for s in K.v.values())
    if ngt == 0 or len(c) == 0: return float('nan')
    o = np.argsort(-c, kind='mergesort'); y = y[o]; ctp = np.cumsum(y); cfp = np.cumsum(1 - y)
    rec = ctp / ngt; prec = ctp / np.maximum(ctp + cfp, 1)
    for i in range(len(prec) - 2, -1, -1): prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side='left'); ap += prec[k] if k < len(prec) else 0.0
    return float(ap / 101)
def klasa_snimke(K, polje='par'):
    """per sequence: total pairs, changes, per confidence bin"""
    R = {}
    for ime, s in K.v.items():
        pr = s[polje]; nr = len(RAZ_KL); a = np.zeros(nr); b = np.zeros(nr)
        for x in pr:
            c1, c2, mc = x[0], x[1], x[2]
            k = next(i for i, (lo, hi) in enumerate(RAZ_KL) if lo <= mc < hi) if mc >= RAZ_KL[0][0] else 0
            a[k] += 1; b[k] += (c1 != c2)
        R[ime] = dict(n=a, sw=b)
    return R
def boot(imena, fn, rng, nboot=NBOOT):
    im = list(imena); vals = []
    for _ in range(nboot):
        smp = [im[i] for i in rng.integers(0, len(im), len(im))]
        v = fn(smp)
        if v == v: vals.append(v)
    if not vals: return (float('nan'), float('nan'))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
def boot2(im1, im2, fn1, fn2, op, rng, nboot=NBOOT):
    a1 = list(im1); a2 = list(im2); vals = []
    for _ in range(nboot):
        s1 = [a1[i] for i in rng.integers(0, len(a1), len(a1))]; s2 = [a2[i] for i in rng.integers(0, len(a2), len(a2))]
        v = op(fn1(s1), fn2(s2))
        if v == v and np.isfinite(v): vals.append(v)
    if not vals: return (float('nan'), float('nan'))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

# ============================================================================ analysis
def analiza():
    rng = np.random.default_rng(SEED)
    P('p67 -- CONFIDENCE AND CALIBRATION OF THE DETECTOR AND CLASS STABILITY (MIPRO 2027)')
    P('detection threshold %.2f; greedy matching by confidence, IoU >= 0.5, class-agnostic; ECE %d bins on [%.2f, 1]; bootstrap over sequences %d x' % (THR, NB, THR, NBOOT))
    SN = {sk: snimke(sk) for sk in SKUPOVI}
    imaju = [(d, sk) for d in DETS for sk in SKUPOVI if SN[sk] and all(json_ok(outp(d, sk, ime)) for ime, _, _ in SN[sk])]
    for d in DETS:
        for sk in SKUPOVI:
            if (d, sk) not in imaju: P('  warning: not all detections for %s x %s -- left out' % (d, NAZ_SK[sk]))
    if not imaju: sys.exit('!! no detections -- first run without --analiza')
    imena_kl = nazivi_klasa(DETS['ssmot'][0])
    REZ = {}; JS = {'postavke': dict(THR=THR, TAU=TAU, NB=NB, NBOOT=NBOOT, RAZ_KL=RAZ_KL), 'kombinacije': {}}
    # reproducibility check against p54
    if ('ssmot', 'ssmot') in imaju and all(os.path.exists(P54 % s) for s in SS16):
        razl = []
        for s in SS16:
            a = json.load(open(outp('ssmot', 'ssmot', 'SSMOT_%d' % s))); b = json.load(open(P54 % s))
            na = sum(len(f['detections']) for f in a); nb_ = sum(len(f['detections']) for f in b)
            ca = sum(x['conf'] for f in a for x in f['detections']); cb = sum(x['conf'] for f in b for x in f['detections'])
            razl.append((na - nb_, ca - cb))
        P('Reproducibility: SSMOT x ssmot versus p54 ds_mux: difference in the number of detections %s; largest difference of the summed confidence %.4f' % (
          'NONE' if all(x[0] == 0 for x in razl) else str([x[0] for x in razl]), max(abs(x[1]) for x in razl)))
    for d, sk in imaju:
        t0 = time.time(); K, info = mjeri(d, sk, SN[sk]); H = hist_snimke(K); im = sorted(H)
        M = mjere_agr(agreg(H, im)); M['AP50'] = ap50(K); M['pomak'] = info['pomak']
        if sk == 'ssmot': M['mjerilo_gt'] = sorted(set(info.get('mjerilo', {}).values()))
        for k, fn in (('ECE', lambda smp: mjere_agr(agreg(H, smp))['ECE']), ('AUROC', lambda smp: mjere_agr(agreg(H, smp))['AUROC']),
                      ('prec_tau', lambda smp: mjere_agr(agreg(H, smp))['prec_tau']), ('tp_iznad_tau', lambda smp: mjere_agr(agreg(H, smp))['tp_iznad_tau']),
                      ('precijenjenost', lambda smp: mjere_agr(agreg(H, smp))['precijenjenost'])):
            M[k + '_ci'] = boot(im, fn, rng)
        A = agreg(H, im); m_ = A['cnt'] > 0
        M['dijagram'] = dict(rubovi=list(np.linspace(THR, 1.0, NB + 1)), broj=A['cnt'].tolist(),
                             preciznost=[float(A['sy'][i] / A['cnt'][i]) if A['cnt'][i] else None for i in range(NB)],
                             srednja_conf=[float(A['sc'][i] / A['cnt'][i]) if A['cnt'][i] else None for i in range(NB)])
        # ACE (equal-mass bins) -- sensitivity
        call = np.concatenate([np.array(K.v[i]['conf']) for i in im]) if im else np.zeros(0)
        if len(call) > NB:
            em = np.unique(np.quantile(call, np.linspace(0, 1, NB + 1))); em[-1] = 1.0 + 1e-9
            M['ACE'] = ece(agreg(hist_snimke(K, eq_mass_edges=em), im))[0]
        # by box size (height / image height) -- descriptive
        hr = np.concatenate([np.array(K.v[i]['hrel'], float) for i in im]) if im else np.zeros(0)
        cc = np.concatenate([np.array(K.v[i]['conf'], float) for i in im]) if im else np.zeros(0)
        yy = np.concatenate([np.array(K.v[i]['y'], int) for i in im]) if im else np.zeros(0, int)
        M['velicina'] = []
        for lo, hi in VEL:
            m = (hr >= lo) & (hr < hi)
            M['velicina'].append(dict(raspon=[lo, hi], n=int(m.sum()), conf=float(cc[m].mean()) if m.any() else None,
                                      preciznost=float(yy[m].mean()) if m.any() else None))
        # class stability
        for polje, ime_m in (('par', 'gt'), ('lanci', 'lanci')):
            KR = klasa_snimke(K, polje)
            npar = sum(KR[i]['n'].sum() for i in im); nsw = sum(KR[i]['sw'].sum() for i in im)
            if npar == 0: continue
            st_fn = lambda smp, KR=KR: (sum(KR[i]['sw'].sum() for i in smp) / max(sum(KR[i]['n'].sum() for i in smp), 1))
            raz = [(float(sum(KR[i]['sw'][k] for i in im)), float(sum(KR[i]['n'][k] for i in im))) for k in range(len(RAZ_KL))]
            M['klasa_' + ime_m] = dict(parova=float(npar), promjena=float(nsw), stopa=float(nsw / npar), ci=boot(im, st_fn, rng),
                                       po_razredu=[(a / b if b else None, b) for a, b in raz])
            if ime_m == 'gt':
                lo_fn = lambda smp, KR=KR: (sum(KR[i]['sw'][0] for i in smp) / max(sum(KR[i]['n'][0] for i in smp), 1)) - \
                                           (sum(KR[i]['sw'][-1] for i in smp) / max(sum(KR[i]['n'][-1] for i in smp), 1))
                M['klasa_gt']['nisko_minus_visoko_ci'] = boot(im, lo_fn, rng)
                parovi = collections.Counter()
                for i in im:
                    for x in K.v[i]['par']:
                        if x[0] != x[1]: parovi[tuple(sorted((x[0], x[1])))] += 1
                M['klasa_gt']['najcesci_parovi'] = [('%s <-> %s' % (imena_kl.get(a, a), imena_kl.get(b, b)), c) for (a, b), c in parovi.most_common(5)]
                tr = [(len(v), len(set(v))) for i in im for v in K.v[i]['tr_kl'].values() if len(v) >= T_MIN_TRAG]
                M['klasa_gt']['tragova'] = len(tr); M['klasa_gt']['tragova_s_promjenom'] = sum(1 for n_, u in tr if u > 1)
                dc = [x[3] for i in im for x in K.v[i]['par']]
                M['kolebanje'] = dict(medijan_abs_dconf=float(np.median(dc)) if dc else None,
                                      tragova_preko_tau=sum(1 for i in im for v in K.v[i]['tr_cf'].values() if len(v) >= T_MIN_TRAG and min(v) < TAU <= max(v)),
                                      tragova=sum(1 for i in im for v in K.v[i]['tr_cf'].values() if len(v) >= T_MIN_TRAG))
        REZ[(d, sk)] = (K, H, im, M); JS['kombinacije']['%s|%s' % (d, sk)] = M
        print('  measured: %s x %s (%.0f s)' % (d, NAZ_SK[sk], time.time() - t0), flush=True)
    # ------------------------------------------------------------------ printout
    f3 = lambda x: '-' if x is None or x != x else '%.3f' % x
    ci = lambda c: '[%s, %s]' % (f3(c[0]), f3(c[1]))
    P(); P('=== A) DETECTOR x SET (threshold 0.25; class-agnostic; IoU >= 0.5)')
    P('  %-6s %-9s %3s %8s %7s %7s %7s %7s %7s %7s %7s %7s %6s' % ('det', 'set', 'off', 'dets', 'prec', 'recall', 'AP50*', 'ECE', 'MCE', 'ACE', 'Brier', 'AUROC', 'scale'))
    for (d, sk), (K, H, im, M) in REZ.items():
        P('  %-6s %-9s %+3d %8d %7s %7s %7s %7s %7s %7s %7s %7s %6s' % (d, NAZ_SK[sk], M['pomak'], M['n'], f3(M['preciznost']), f3(M['odziv']), f3(M['AP50']),
          f3(M['ECE']), f3(M['MCE']), f3(M.get('ACE')), f3(M['brier']), f3(M['AUROC']), ','.join('%.1f' % x for x in M.get('mjerilo_gt', [])) or '-'))
    P('  * class-agnostic AP50, cut at confidence >= 0.25 (not comparable with the mAP from the literature)')
    P(); P('=== B) CALIBRATION AND THRESHOLD tau = 0.8 (95 %% bootstrap over sequences)')
    for (d, sk), (K, H, im, M) in REZ.items():
        P('  %-6s %-9s ECE %s %s | overconfidence %s %s | AUROC %s %s | prec(p>=0.8) %s %s | share of TP with p>=0.8 %s %s' % (
          d, NAZ_SK[sk], f3(M['ECE']), ci(M['ECE_ci']), f3(M['precijenjenost']), ci(M['precijenjenost_ci']), f3(M['AUROC']), ci(M['AUROC_ci']),
          f3(M['prec_tau']), ci(M['prec_tau_ci']), f3(M['tp_iznad_tau']), ci(M['tp_iznad_tau_ci'])))
    P(); P('=== C) RELIABILITY DIAGRAM (bin: count, mean confidence -> precision)')
    for (d, sk), (K, H, im, M) in REZ.items():
        dg = M['dijagram']
        P('  %-6s %-9s ' % (d, NAZ_SK[sk]) + ' '.join('%.2f:%d %.2f>%.2f' % (dg['rubovi'][i], dg['broj'][i], dg['srednja_conf'][i], dg['preciznost'][i])
                                                    for i in range(NB) if dg['broj'][i]))
    P(); P('=== C2) BY BOX SIZE (height / image height): count, mean confidence, precision -- descriptive')
    for (d, sk), (K, H, im, M) in REZ.items():
        P('  %-6s %-9s ' % (d, NAZ_SK[sk]) + ' | '.join('%s-%s %%: n %d, p %s, prec %s' % (('%g' % (100 * v['raspon'][0])), ('%g' % (100 * min(v['raspon'][1], 1.0))), v['n'], f3(v['conf']), f3(v['preciznost'])) for v in M['velicina']))
    P(); P('=== D) CLASS STABILITY (meaningful for the ssmot detector; coco must give 0 -- check of the class filter)')
    for (d, sk), (K, H, im, M) in REZ.items():
        for ime_m, opis in (('gt', 'with GT (the same GT track in consecutive frames)'), ('lanci', 'without GT (chains of detections, IoU >= 0.5)')):
            k = M.get('klasa_' + ime_m)
            if not k: continue
            pr = ', '.join('[%.2f-%.2f) %s (n %d)' % (RAZ_KL[i][0], min(RAZ_KL[i][1], 1.0), '-' if r is None else '%.2f %%' % (100 * r), n_) for i, (r, n_) in enumerate(k['po_razredu']))
            P('  %-6s %-9s %-44s rate %.2f %% %s, pairs %d | by confidence: %s' % (d, NAZ_SK[sk], opis, 100 * k['stopa'],
              '[%.2f, %.2f] %%' % (100 * k['ci'][0], 100 * k['ci'][1]), k['parova'], pr))
            if ime_m == 'gt':
                P('  %-16s tracks (>= %d frames) %d, with at least one change %d; low - high %s; most common pairs: %s' % (
                  '', T_MIN_TRAG, k['tragova'], k['tragova_s_promjenom'], '[%.2f, %.2f] %%' % (100 * k['nisko_minus_visoko_ci'][0], 100 * k['nisko_minus_visoko_ci'][1]),
                  '; '.join('%s %d' % x for x in k['najcesci_parovi']) or '-'))
                kb = M['kolebanje']
                P('  %-16s fluctuation: median |dp| %s; tracks crossing 0.8: %d of %d' % ('', f3(kb['medijan_abs_dconf']), kb['tragova_preko_tau'], kb['tragova']))
    # ------------------------------------------------------------------ expectations
    P(); P('=== E) EXPECTATIONS SET IN ADVANCE')
    E = {}
    def fnm(H, key): return lambda smp: mjere_agr(agreg(H, smp))[key]
    if ('ssmot', 'ssmot') in REZ:
        Hs, ims = REZ[('ssmot', 'ssmot')][1], REZ[('ssmot', 'ssmot')][2]
        for sk in ('smd', 'fvessel'):
            if ('ssmot', sk) not in REZ: continue
            Ho, imo = REZ[('ssmot', sk)][1], REZ[('ssmot', sk)][2]
            c = boot2(imo, ims, fnm(Ho, 'ECE'), fnm(Hs, 'ECE'), lambda a, b: a - b, rng)
            dd = REZ[('ssmot', sk)][3]['ECE'] - REZ[('ssmot', 'ssmot')][3]['ECE']
            E['E1_' + sk] = dict(razlika=dd, ci=c, potvrdeno=bool(c[0] > 0))
            P('  E1 ssmot: ECE %s - SSMOT = %+.3f %s -> %s' % (NAZ_SK[sk], dd, ci(c), 'CONFIRMED' if c[0] > 0 else ('OPPOSITE' if c[1] < 0 else 'not confirmed')))
    ok2 = []
    for (d, sk), (K, H, im, M) in REZ.items():
        ok2.append(M['AUROC_ci'][0] > 0.5)
        P('  E2 %-6s %-9s AUROC %s %s -> %s' % (d, NAZ_SK[sk], f3(M['AUROC']), ci(M['AUROC_ci']), 'above 0.5' if M['AUROC_ci'][0] > 0.5 else 'NOT above 0.5'))
    E['E2'] = dict(svih=len(ok2), potvrdeno=int(sum(ok2)))
    if ('ssmot', 'ssmot') in REZ and ('ssmot', 'smd') in REZ and REZ[('ssmot', 'ssmot')][3].get('klasa_gt') and REZ[('ssmot', 'smd')][3].get('klasa_gt'):
        Ks, Ko = REZ[('ssmot', 'ssmot')][0], REZ[('ssmot', 'smd')][0]
        KRs, KRo = klasa_snimke(Ks), klasa_snimke(Ko)
        fs = lambda smp: sum(KRs[i]['sw'].sum() for i in smp) / max(sum(KRs[i]['n'].sum() for i in smp), 1)
        fo = lambda smp: sum(KRo[i]['sw'].sum() for i in smp) / max(sum(KRo[i]['n'].sum() for i in smp), 1)
        c = boot2(sorted(KRo), sorted(KRs), fo, fs, lambda a, b: a / b if b > 0 else float('nan'), rng)
        om = REZ[('ssmot', 'smd')][3]['klasa_gt']['stopa'] / max(REZ[('ssmot', 'ssmot')][3]['klasa_gt']['stopa'], 1e-12)
        E['E3'] = dict(omjer=om, ci=c, potvrdeno=bool(om >= 3 and c[0] > 1))
        P('  E3 ratio of the class change rate SMD-Plus / SSMOT = %.1f x %s -> %s' % (om, '[%.1f, %.1f]' % c, 'CONFIRMED' if (om >= 3 and c[0] > 1) else 'not confirmed'))
    for sk in ('ssmot', 'smd'):
        if ('ssmot', sk) in REZ and REZ[('ssmot', sk)][3].get('klasa_gt'):
            k = REZ[('ssmot', sk)][3]['klasa_gt']; c = k['nisko_minus_visoko_ci']
            E['E4_' + sk] = dict(ci=c, potvrdeno=bool(c[0] > 0))
            P('  E4 %-9s change rate [0.25-0.5) minus [0.8-1]: %s -> %s' % (NAZ_SK[sk], '[%.2f, %.2f] %%' % (100 * c[0], 100 * c[1]), 'CONFIRMED' if c[0] > 0 else 'not confirmed'))
    P('  E5 (descriptive) share of TP with confidence >= 0.8: ' + '; '.join('%s/%s %s' % (d, NAZ_SK[sk], f3(M['tp_iznad_tau'])) for (d, sk), (_, _, _, M) in REZ.items()))
    JS['ocekivanja'] = E
    # ------------------------------------------------------------------ sensitivity
    P(); P('=== F) SENSITIVITY (ECE / AUROC / class change rate with GT)')
    for (d, sk), (K, H, im, M) in REZ.items():
        rows = []
        K3, _ = mjeri(d, sk, SN[sk], thr=0.3, s_lancima=False); M3 = mjere_agr(agreg(hist_snimke(K3), sorted(K3.v)))
        rows.append('IoU 0,3: ECE %s AUROC %s' % (f3(M3['ECE']), f3(M3['AUROC'])))
        if sk == 'smd':
            Kb, _ = mjeri(d, sk, SN[sk], bez=DVOSMISLENE, s_lancima=False); Mb = mjere_agr(agreg(hist_snimke(Kb), sorted(Kb.v)))
            kr = klasa_snimke(Kb); st_ = sum(kr[i]['sw'].sum() for i in kr) / max(sum(kr[i]['n'].sum() for i in kr), 1)
            rows.append('without MVI_1584/1615: ECE %s AUROC %s class %.2f %%' % (f3(Mb['ECE']), f3(Mb['AUROC']), 100 * st_))
        if sk in ('ssmot', 'smd'):
            for dp in (-1, 1):
                Kp, ip = mjeri(d, sk, SN[sk], pomak_sila=M['pomak'] + dp, s_lancima=False); Mp = mjere_agr(agreg(hist_snimke(Kp), sorted(Kp.v)))
                kr = klasa_snimke(Kp); st_ = sum(kr[i]['sw'].sum() for i in kr) / max(sum(kr[i]['n'].sum() for i in kr), 1)
                rows.append('offset %+d: ECE %s class %.2f %%' % (M['pomak'] + dp, f3(Mp['ECE']), 100 * st_))
        P('  %-6s %-9s %s' % (d, NAZ_SK[sk], ' | '.join(rows)))
    st = stanje()
    if st.get('vrijeme'):
        P(); P('=== G) DETECTION SPEED (whole sequence, with start-up)')
        for d in DETS:
            for sk in SKUPOVI:
                v = [x for k, x in st['vrijeme'].items() if k.startswith('%s|%s|' % (d, sk))]
                if v: P('  %-6s %-9s %.0f FPS' % (d, NAZ_SK[sk], sum(x[0] for x in v) / sum(x[1] for x in v)))
    P(); P('Notes: threshold 0.25 (calibration only on [0.25, 1]); FVessel GT 1 Hz and incomplete (small boats without a label); SMD IDs derived (p63);')
    P('          SMD distractors (Buoy, Other) left out; SSMOT frames are not published.')
    json.dump(JS, open(OUT_JSON, 'w'), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    open(OUT_TXT, 'w').write('\n'.join(L) + '\n')
    print('\nresult: %s\ndata for the figures: %s' % (OUT_TXT, OUT_JSON))

# ============================================================================ FVessel: same offset for both detectors (p67b)
FV_GLAVNI = [0, 12, 21]          # 0 = nominal (frame = round(s * fps)); 12 and 21 = offsets that p67 chose for ssmot and coco
FV_MREZA = list(range(-24, 49, 6)) + [-1, 1, 11, 13, 20, 22]
N_TAU_MIN = 100                  # precision above 0.8 is reported only with at least 100 detections above the threshold
def fv_pomak():
    """p67b (plan written before the run): FVessel with the SAME offset for both detectors; the offset is not chosen by IoU."""
    rng = np.random.default_rng(SEED); out = []
    def Q(x=''): print(x, flush=True); out.append(str(x))
    SN = {sk: snimke(sk) for sk in ('ssmot', 'fvessel')}
    for d in DETS:
        for sk in SN:
            if not all(json_ok(outp(d, sk, ime)) for ime, _, _ in SN[sk]): sys.exit('!! no detections %s x %s' % (d, sk))
    Q('p67b -- FVessel with the same offset for both detectors')
    Q('main offset = 0 (nominal, frame = round(s x fps)); sensitivity 12 and 21; P(p>=0.8) only with >= %d detections above 0.8' % N_TAU_MIN)
    Ks, _ = mjeri('ssmot', 'ssmot', SN['ssmot'], s_lancima=False); Hs = hist_snimke(Ks); ims = sorted(Hs)
    f3 = lambda x: '-' if x is None or x != x else '%.3f' % x
    def red(d, k, ci=False):
        K, _ = mjeri(d, 'fvessel', SN['fvessel'], pomak_sila=k, s_lancima=False); H = hist_snimke(K); im = sorted(H); A = agreg(H, im); M = mjere_agr(A)
        hr = np.concatenate([np.array(K.v[i]['hrel'], float) for i in im]); cc = np.concatenate([np.array(K.v[i]['conf'], float) for i in im])
        yy = np.concatenate([np.array(K.v[i]['y'], int) for i in im]); big = hr >= 0.10
        r = dict(k=k, P=M['preciznost'], R=M['odziv'], ECE=M['ECE'], delta=M['precijenjenost'], conf=float(cc.mean()) if len(cc) else float('nan'),
                 AUROC=M['AUROC'], n_tau=int(A['tau_n']), P_tau=M['prec_tau'], cov=M['tp_iznad_tau'], n=int(A['n']),
                 big_udio=float(big.mean()) if len(big) else float('nan'), big_conf=float(cc[big].mean()) if big.any() else float('nan'),
                 big_P=float(yy[big].mean()) if big.any() else float('nan'))
        if ci:
            fn = lambda key: (lambda smp: mjere_agr(agreg(H, smp))[key])
            r['ECE_ci'] = boot(im, fn('ECE'), rng); r['AUROC_ci'] = boot(im, fn('AUROC'), rng); r['delta_ci'] = boot(im, fn('precijenjenost'), rng)
            if d == 'ssmot':
                r['dECE'] = M['ECE'] - mjere_agr(agreg(Hs, ims))['ECE']
                r['dECE_ci'] = boot2(im, ims, fn('ECE'), lambda smp: mjere_agr(agreg(Hs, smp))['ECE'], lambda a, b: a - b, rng)
        return r
    R = {}
    for d in DETS:
        Q(); Q('=== %s (%s)' % (d, DETS[d][2]))
        Q('  %4s %6s %6s %6s %7s %6s %6s %6s %8s %6s %6s | large boxes (>= 10 %% of the height): share, p, P' % ('k', 'dets', 'P', 'R', 'ECE', 'delta', 'mean.p', 'AUROC', 'n(p>=.8)', 'P>=.8', 'cov.'))
        for k in sorted(set(FV_MREZA + FV_GLAVNI)):
            r = red(d, k, ci=k in FV_GLAVNI); R[(d, k)] = r
            Q('  %+4d %6d %6s %6s %7s %+6.3f %6s %6s %8d %6s %6s | %s %s %s%s' % (k, r['n'], f3(r['P']), f3(r['R']), f3(r['ECE']), r['delta'], f3(r['conf']), f3(r['AUROC']),
              r['n_tau'], f3(r['P_tau']) if r['n_tau'] >= N_TAU_MIN else 'malo', f3(r['cov']), f3(r['big_udio']), f3(r['big_conf']), f3(r['big_P']),
              '  <- main' if k == 0 else ('  <- sensitivity' if k in FV_GLAVNI else '')))
    Q(); Q('=== MAIN OFFSETS with 95 % bootstrap intervals')
    for d in DETS:
        for k in FV_GLAVNI:
            r = R[(d, k)]
            Q('  %-6s k %+3d  ECE %s [%s, %s]  delta %+.3f [%+.3f, %+.3f]  AUROC %s [%s, %s]%s' % (d, k, f3(r['ECE']), f3(r['ECE_ci'][0]), f3(r['ECE_ci'][1]),
              r['delta'], r['delta_ci'][0], r['delta_ci'][1], f3(r['AUROC']), f3(r['AUROC_ci'][0]), f3(r['AUROC_ci'][1]),
              ('  | ECE FVessel - SSMOT %+.3f [%+.3f, %+.3f]' % (r['dECE'], r['dECE_ci'][0], r['dECE_ci'][1])) if d == 'ssmot' else ''))
    Q(); Q('=== RULES SET IN ADVANCE (p67b)')
    e1 = all(R[('ssmot', k)]['dECE_ci'][0] > 0 for k in FV_GLAVNI)
    pre = all(R[('ssmot', k)]['delta_ci'][0] > 0 for k in FV_GLAVNI)
    Q('  E1 (river): ECE of ssmot on FVessel higher than on SSMOT at ALL THREE offsets -> %s' % ('ROBUST' if e1 else 'NOT ROBUST (do not claim)'))
    Q('  overconfidence of ssmot on FVessel (delta > 0) at all three offsets -> %s' % ('ROBUST' if pre else 'NOT ROBUST'))
    for d in DETS:
        lo = [R[(d, k)]['AUROC_ci'][0] for k in FV_GLAVNI]
        ocj = 'not computable (no TP or FP)' if any(x != x for x in lo) else ('separated from chance' if min(lo) > 0.55 else 'practically not separated from chance (lower bound <= 0.55)')
        Q('  AUROC %-6s lower bound at 0/12/21: %s -> %s' % (d, ' / '.join(f3(x) for x in lo), ocj))
    open(S + '/p67_fvessel_pomak.txt', 'w').write('\n'.join(out) + '\n')
    print('\nresult: %s/p67_fvessel_pomak.txt' % S)

# ============================================================================ p67c: calibration against IoU (LaECE0, Kuzucu et al. ECCV 2024)
def laece():
    """p67c (added later, plan written before the run): confidence target = IoU with the matched GT vessel (greedy matching by confidence,
    IoU > 0), 0 for the unmatched; distractors left out. Comparison with ECE with a binary hit at IoU 0.5 on the same detections."""
    rng = np.random.default_rng(SEED); out = []
    def Q(x=''): print(x, flush=True); out.append(str(x))
    Q('p67c -- LaECE0 (target = IoU) versus ECE@0.5 (target = hit); SSMOT and SMD-Plus, section p67c')
    fine = np.linspace(THR, 1.0, NB + 1); fine[-1] = 1.0 + 1e-9
    R = {}
    for d in DETS:
        for sk in ('ssmot', 'smd'):
            SN = snimke(sk)
            if not all(json_ok(outp(d, sk, ime)) for ime, _, _ in SN): Q('  no detections %s x %s' % (d, sk)); continue
            H = {}; ious = []
            for ime, v, g in SN:
                Dt = ucitaj_det(outp(d, sk, ime)); mw, mh, fps, n = meta(v)
                if sk == 'ssmot': Gv, Gd, _ = gt_ssmot(g, Dt)
                else: Gv, Gd = gt_smd(g)
                C = []; Tl = []; C5 = []; Y5 = []
                for t in sorted(set(range(1, n + 1)) | set(Gv)):
                    dl = Dt.get(t, [])
                    if not dl: continue
                    for (c_, p_, y_, gid_, hb_) in upari(dl, Gv.get(t, []), [b_ for _, b_ in Gd.get(t, [])], 0.5):
                        C5.append(p_); Y5.append(y_)
                    G = [(b_, True) for _, b_ in Gv.get(t, [])] + [(b_, False) for _, b_ in Gd.get(t, [])]
                    used = [False] * len(G)
                    for (c, b, p) in sorted(dl, key=lambda x: -x[2]):
                        bi, bj = 0.0, -1
                        for j, (gb, pl) in enumerate(G):
                            if used[j]: continue
                            vv = iou(b, gb)
                            if vv > bi: bi, bj = vv, j
                        if bj < 0: C.append(p); Tl.append(0.0); continue
                        used[bj] = True
                        if not G[bj][1]: continue                     # distractor
                        C.append(p); Tl.append(bi)
                        if bi >= 0.5: ious.append(bi)
                C = np.array(C); Tl = np.array(Tl); C5 = np.array(C5); Y5 = np.array(Y5, float)
                bb = np.clip(np.digitize(C, fine) - 1, 0, NB - 1); b5 = np.clip(np.digitize(C5, fine) - 1, 0, NB - 1)
                H[ime] = dict(cnt=np.bincount(bb, minlength=NB).astype(float), sc=np.bincount(bb, weights=C, minlength=NB),
                              st=np.bincount(bb, weights=Tl, minlength=NB), cnt5=np.bincount(b5, minlength=NB).astype(float),
                              sc5=np.bincount(b5, weights=C5, minlength=NB), sy=np.bincount(b5, weights=Y5, minlength=NB))
            def mj(smp, key):
                k5 = key == 'sy'
                cnt = sum(H[i]['cnt5' if k5 else 'cnt'] for i in smp); sc = sum(H[i]['sc5' if k5 else 'sc'] for i in smp)
                tg = sum(H[i][key] for i in smp); N = cnt.sum(); m = cnt > 0
                return float((np.abs(tg[m] - sc[m])).sum() / N), float((sc.sum() - tg.sum()) / N)
            im = sorted(H)
            la, dla = mj(im, 'st'); e5, d5 = mj(im, 'sy')
            R[(d, sk)] = dict(LaECE=la, dLa=dla, ECE=e5, d5=d5, la_ci=boot(im, lambda smp: mj(smp, 'st')[0], rng), e_ci=boot(im, lambda smp: mj(smp, 'sy')[0], rng),
                              mIoU=float(np.mean(ious)) if ious else float('nan'))
            r = R[(d, sk)]
            Q('  %-6s %-9s LaECE0 %.3f [%.3f, %.3f] (mean p - mean IoU %+.3f) | ECE@0.5 %.3f [%.3f, %.3f] (delta %+.3f) | mean IoU of the hits %.3f' % (
              d, NAZ_SK[sk], r['LaECE'], r['la_ci'][0], r['la_ci'][1], r['dLa'], r['ECE'], r['e_ci'][0], r['e_ci'][1], r['d5'], r['mIoU']))
    Q(); Q('=== RULE SET IN ADVANCE (p67c)')
    ok = [k for k, r in R.items() if r['LaECE'] <= r['ECE'] / 2]
    Q('  LaECE0 <= ECE@0.5 / 2 in %d of %d combinations -> %s' % (len(ok), len(R), 'the underconfidence is MOSTLY a consequence of the binary target (confidence follows IoU)'
      if len(ok) >= 3 else 'the explanation via IoU is NOT confirmed'))
    Q('  (ECE@0.5 here must equal the main p67 printout -- check)')
    open(S + '/p67_laece.txt', 'w').write('\n'.join(out) + '\n'); print('\nresult: %s/p67_laece.txt' % S)

# ============================================================================ p68: threshold from a mixture of two Gaussian distributions (without labels)
def gmm2(z, inits=((0.10, 0.90), (0.25, 0.75), (0.40, 0.60)), it_max=2000, tol=1e-9):
    """EM for a 1-D mixture of two Gaussian distributions; several deterministic starts, returns the best log-likelihood."""
    z = np.asarray(z, float); best = None; sd0 = max(z.std(), 1e-3)
    for qa, qb in inits:
        w = np.array([0.5, 0.5]); mu = np.quantile(z, [qa, qb]).astype(float); sd = np.array([sd0 / 2, sd0 / 2]); ll_old = -np.inf
        for _ in range(it_max):
            dens = w / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((z[:, None] - mu) / sd) ** 2) + 1e-300
            tot = dens.sum(1); ll = np.log(tot).sum(); r = dens / tot[:, None]
            nk = r.sum(0) + 1e-12; w = nk / len(z); mu = (r * z[:, None]).sum(0) / nk
            sd = np.sqrt(np.maximum((r * (z[:, None] - mu) ** 2).sum(0) / nk, 1e-4))
            if ll - ll_old < tol * abs(ll): break
            ll_old = ll
        if best is None or ll > best[3]: best = (w.copy(), mu.copy(), sd.copy(), ll)
    w, mu, sd, ll = best; o = np.argsort(mu)
    return w[o], mu[o], sd[o], ll                                   # component 1 = high confidence

def p2z(p): u = np.clip((np.asarray(p, float) - THR) / (1 - THR), 1e-4, 1 - 1e-4); return np.log(u / (1 - u))
def z2p(z): return THR + (1 - THR) / (1 + np.exp(-z))

def gmm_prag(w, mu, sd, zmin=None, zmax=None):
    """threshold = the point where the posterior probability of the high component crosses 0.5 (from < 0.5 to >= 0.5, increasing z), within the range
    of the data; if there are several such points (unequal variances), the one closest to the midpoint of the means is taken; None if there is none"""
    lo = mu[0] - 4 * sd[0] if zmin is None else zmin; hi = mu[1] + 4 * sd[1] if zmax is None else zmax
    g = np.linspace(lo, hi, 20001)
    d = w / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((g[:, None] - mu) / sd) ** 2)
    post = d[:, 1] / (d.sum(1) + 1e-300); up = np.where((post[:-1] < 0.5) & (post[1:] >= 0.5))[0]
    if not len(up): return None
    k = up[np.argmin(np.abs(g[up] - (mu[0] + mu[1]) / 2))] + 1
    return float(z2p(g[k]))

GMM_PREC_MIN = 0.90      # rule p68 (written before the run): precision above the GMM threshold >= 0.90 ...
GMM_D_ODV = 2.0          # ... and Ashman's D > 2 = cleanly separated components
def gmm_mod():
    """p68 (plan written before the run): fast confirmation threshold from the confidence distribution
    WITHOUT labels. Mixture of two Gaussian distributions on the logit of the confidence (all frames, all detections) on the FIRST half of the sequences of the set;
    evaluation with labels on the SECOND half: precision and coverage above the GMM threshold versus the fixed threshold 0.8 and versus an 'oracle'
    (smallest threshold with precision >= 0.95, computed with labels on the same sequences). Offset 0 on all sets."""
    out = []
    def Q(x=''): print(x, flush=True); out.append(str(x))
    f3 = lambda x: '-' if x is None or x != x else '%.3f' % x
    Q('p68 -- confidence threshold from a mixture of two Gaussian distributions without labels')
    Q('fit: logit((p - 0.25) / 0.75), all detections of all frames of the FIRST half of the sequences; evaluation: SECOND half, IoU >= 0.5, offset 0')
    def ocjena(C, Y, tau):
        if tau is None: return dict(n=0, P=float('nan'), cov=float('nan'))
        m = C >= tau; ntp = Y.sum()
        return dict(n=int(m.sum()), P=float(Y[m].mean()) if m.any() else float('nan'), cov=float(Y[m].sum() / ntp) if ntp else float('nan'))
    def prorok(C, Y):
        for t in np.round(np.arange(THR, 0.995, 0.01), 2):
            m = C >= t
            if m.sum() >= N_TAU_MIN and Y[m].mean() >= 0.95: return float(t)
        return None
    R = {}
    for d in DETS:
        for sk in SKUPOVI:
            SN = snimke(sk)
            if not SN or not all(json_ok(outp(d, sk, ime)) for ime, _, _ in SN): Q('  no detections %s x %s' % (d, sk)); continue
            h = len(SN) // 2; kal = [s for s in SN[:h]]; evl = [s for s in SN[h:]]
            def sirove(lst):
                return np.array([p for ime, _, _ in lst for fl in ucitaj_det(outp(d, sk, ime)).values() for (_, _, p) in fl], float)
            K, info = mjeri(d, sk, SN, pomak_sila=0, s_lancima=False)
            def oznake(lst):
                C = np.concatenate([np.array(K.v[ime]['conf'], float) for ime, _, _ in lst if ime in K.v] or [np.zeros(0)])
                Y = np.concatenate([np.array(K.v[ime]['y'], int) for ime, _, _ in lst if ime in K.v] or [np.zeros(0, int)])
                return C, Y
            for nacin, fit_lst, ev_lst in (('pola', kal, evl), ('sve', SN, SN)):
                pc = sirove(fit_lst)
                if len(pc) < 50: Q('  too few detections for the fit %s x %s (%s)' % (d, sk, nacin)); continue
                w, mu, sd, ll = gmm2(p2z(pc)); D = float(np.sqrt(2) * abs(mu[1] - mu[0]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2))
                zz = p2z(pc); tau = gmm_prag(w, mu, sd, zz.min(), zz.max()); C, Y = oznake(ev_lst); tp = prorok(C, Y)
                R[(d, sk, nacin)] = dict(w=w, mu=mu, sd=sd, D=D, tau=tau, n_fit=len(pc), n_ev=len(C), g=ocjena(C, Y, tau), f=ocjena(C, Y, TAU),
                                         tp=tp, o=ocjena(C, Y, tp))
    for nacin, nas in (('pola', 'MAIN: fit on the first half of the sequences, evaluation on the second'), ('sve', 'DESCRIPTIVE: fit and evaluation on all sequences')):
        Q(); Q('=== %s' % nas)
        Q('  %-6s %-9s %7s %6s %6s %6s | %-24s | %-24s | %-24s' % ('det', 'set', 'n_fit', 'D', 'w_vis', 'tau', 'GMM: n / P / cov.', 'fixed 0.8: n / P / cov.', 'oracle P>=0.95: tau / cov.'))
        for d in DETS:
            for sk in SKUPOVI:
                r = R.get((d, sk, nacin))
                if not r: continue
                Q('  %-6s %-9s %7d %6.2f %6.2f %6s | %6d / %s / %s | %6d / %s / %s | %6s / %s' % (d, NAZ_SK[sk], r['n_fit'], r['D'], r['w'][1], f3(r['tau']),
                  r['g']['n'], f3(r['g']['P']), f3(r['g']['cov']), r['f']['n'], f3(r['f']['P']), f3(r['f']['cov']), f3(r['tp']), f3(r['o']['cov'])))
                Q('  %17s components (in confidence, mean): low %.2f, high %.2f' % ('', float(z2p(r['mu'][0])), float(z2p(r['mu'][1]))))
    Q(); Q('=== RULES SET IN ADVANCE (p68; evaluation on the second half of the sequences)')
    luka = [(d, sk) for d in DETS for sk in ('ssmot', 'smd') if (d, sk, 'pola') in R]
    ok = []
    for d, sk in luka:
        r = R[(d, sk, 'pola')]; uvjet = r['tau'] is not None and r['g']['P'] >= GMM_PREC_MIN and r['g']['cov'] >= r['f']['cov']
        ok.append(uvjet)
        Q('  R1 %-6s %-9s P(GMM) %s >= %.2f and cov.(GMM) %s >= cov.(0.8) %s -> %s' % (d, NAZ_SK[sk], f3(r['g']['P']), GMM_PREC_MIN, f3(r['g']['cov']), f3(r['f']['cov']), 'DA' if uvjet else 'NE'))
    Q('  R1 total: %d of %d -> %s' % (sum(ok), len(ok), 'the label-free GMM threshold REPLACES the fixed threshold 0.8 (precision >= 0.90, coverage not lower)'
      if len(ok) == 4 and sum(ok) >= 3 else 'NOT confirmed'))
    if all((d, sk, 'pola') in R for d, sk in luka) and ('ssmot', 'fvessel', 'pola') in R:
        Dl = [R[(d, sk, 'pola')]['D'] for d, sk in luka]; Dfv = R[('ssmot', 'fvessel', 'pola')]['D']
        flag = Dfv <= GMM_D_ODV and all(x > GMM_D_ODV for x in Dl)
        Q('  R2 Ashman\'s D: port/coast %s; ssmot on FVessel %.2f -> %s' % (' / '.join('%.2f' % x for x in Dl), Dfv,
          'D FLAGS an out-of-domain detector (FVessel <= 2, all others > 2)' if flag else 'flag NOT confirmed'))
        if ('coco', 'fvessel', 'pola') in R: Q('  (descriptive) coco on FVessel: D %.2f' % R[('coco', 'fvessel', 'pola')]['D'])
    open(S + '/p68_gmm.txt', 'w').write('\n'.join(out) + '\n'); print('\nresult: %s/p68_gmm.txt' % S)

if __name__ == '__main__':
    if '--gmm' in sys.argv: gmm_mod(); sys.exit(0)
    if '--laece' in sys.argv: laece(); sys.exit(0)
    if '--fvessel-pomaci' in sys.argv: fv_pomak(); sys.exit(0)
    if '--vrati' in sys.argv:
        os.makedirs(O, exist_ok=True)
        if alive(LOCK): sys.exit('!! p67 is still running (PID %s).' % alive(LOCK))
        vrati(); print('done.'); sys.exit(0)
    if '--analiza' not in sys.argv: runs()
    analiza()
