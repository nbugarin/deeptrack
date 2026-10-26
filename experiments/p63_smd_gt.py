#!/usr/bin/env python3
"""p63 -- SMD-Plus VIS_Onshore -> set for TRACKING evaluation (without manual annotation).

SMD-Plus has a box and a class for EVERY frame, but no track ID (p62). The ID is derived automatically:
linking boxes from frame to frame by overlap with the predicted position (constant velocity; Hungarian algorithm,
IoU >= 0.4; bridging gaps of up to 1 s with IoU >= 0.3). p62 showed that 98.9 % of pairs of neighbouring frames are unambiguous, so the derivation error is small;
ambiguous places are counted and printed, and for 3 sequences check videos are made so that the IDs can be checked by eye.

Classes: vessels are evaluated (Ferry, Vessel/ship, Boat, Kayak, Sail boat); Buoy and Other are "distractors" (as in MOT17):
neither a hit nor an error -- a tracker box on them is not counted as false.

Parts:
 A) derivation of the IDs + statistics per sequence  ->  /workspace/SMD_mot/<sequence>/gt/gt.txt
    (MOT: frame, id, x, y, w, h, 1, SMD_class, 1; frame 1 = first frame of the video)
 B) check videos with IDs (960x540) for 3 sequences  ->  /workspace/SMD_mot/kontrola/*.mp4
 C) re-encoding of the videos to H.264 mp4 (as for FVessel)  ->  /workspace/SMD_mot/<sequence>/<sequence>.mp4
 D) quick detector check (if the GPU is free): SSMOT 640/1280, COCO boat 640/1280 on every 30th frame

Run:            python3 /workspace/p63_smd_gt.py /workspace/SMD_Plus/VIS_Onshore
Only D, later:  python3 /workspace/p63_smd_gt.py /workspace/SMD_Plus/VIS_Onshore --samo-detektor
Output:         /workspace/scpilot/p63_smd_gt.txt
"""
import os, sys, glob, re, json, time, subprocess, collections
import numpy as np

W = os.environ.get('P60_W', '/workspace')
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
ROOT = ARGS[0] if ARGS else W + '/SMD_Plus/VIS_Onshore'
SAMO_D = '--samo-detektor' in sys.argv
OD = W + '/SMD_mot'; OUT = W + '/scpilot/p63_smd_gt.txt'
PLOVILA = {'ferry', 'vessel/ship', 'boat', 'kayak', 'sail boat', 'sailboat', 'speed boat'}
IOU_LINK, IOU_GAP, GAP = 0.4, 0.3, 30
KONTROLA = ['MVI_1584_VIS', 'MVI_1474_VIS', 'MVI_1470_VIS']
os.makedirs(OD, exist_ok=True); os.makedirs(os.path.dirname(OUT), exist_ok=True)
L = []
def P(x=''):
    print(x, flush=True); L.append(str(x))
def spremi():
    open(OUT if not SAMO_D else OUT.replace('.txt', '_detektor.txt'), 'w').write('\n'.join(L) + '\n')
import scipy.io as sio
from scipy.optimize import linear_sum_assignment
try:
    import cv2
except ImportError:
    cv2 = None

# ----------------------------------------------------------------------------- loading
if not os.path.isdir(ROOT): sys.exit('no folder %s' % ROOT)
vid = sorted(p for e in ('avi', 'mp4', 'mov', 'AVI', 'MP4', 'MOV') for p in glob.glob(ROOT + '/**/*.' + e, recursive=True))
mats = sorted(glob.glob(ROOT + '/**/*.mat', recursive=True))
def kljuc(p): return re.sub(r'[_-]?ObjectGT$', '', os.path.splitext(os.path.basename(p))[0], flags=re.I)
VK = {kljuc(v): v for v in vid}
SN = [(kljuc(m), m, VK[kljuc(m)]) for m in mats if kljuc(m) in VK]
if not SN: sys.exit('no matched .mat + video in %s' % ROOT)

def ucitaj(m):
    d = sio.loadmat(m, squeeze_me=True, struct_as_record=False)
    S = d['structXML'].ravel(); fr = []
    for e in S:
        b = np.asarray(getattr(e, 'BB'), dtype=float)
        b = np.zeros((0, 4)) if b.size == 0 else np.atleast_2d(b).reshape(-1, 4)
        n = len(b)
        def niz(f):
            if not hasattr(e, f): return [None] * n
            v = np.atleast_1d(np.asarray(getattr(e, f), dtype=object)).ravel().tolist()
            return (v + [None] * n)[:n]
        fr.append((b, niz('Object'), niz('ObjectType'), niz('MotionType')))
    return fr
def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[0] + a[2], b[0] + b[2]); y2 = min(a[1] + a[3], b[1] + b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1); u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0

def ucitaj_gt_mot(p):
    g = collections.defaultdict(list)
    for l in open(p):
        s = l.strip().split(',')
        if len(s) >= 8: g[int(s[0])].append((int(s[1]), [float(q) for q in s[2:6]], int(s[7])))
    return g

KLASE = {}
if not SAMO_D:
    # ========================================================================= A. IDs
    P('p63 SMD-Plus VIS_Onshore -> MOT, %d sequences; linking IoU >= %.1f, bridging up to %d frames with IoU >= %.1f' % (len(SN), IOU_LINK, GAP, IOU_GAP))
    P(); P('=== A. DERIVED IDs PER SEQUENCE')
    P('  %-18s %6s %7s %5s %8s %7s %6s %7s %6s' % ('sequence', 'frames', 'boxes', 'ID', 'track fr', 'ambig.', 'bridg.', 'cls.chg', 'distr.'))
    UK = collections.Counter(); DV = []
    for ime, m, v in SN:
        F = ucitaj(m)
        for b, oc, ot, mt in F:
            for c, t in zip(oc, ot):
                if c is not None: KLASE[int(c)] = str(t)
        akt = {}; nid = 0; rows = []; dv = 0; prem = 0; kp = 0; duz = collections.Counter()
        def pred(k, t):                                   # constant velocity: last box + v * elapsed time
            lb, lf, lc, vv = akt[k]
            return [lb[0] + vv[0] * (t - lf), lb[1] + vv[1] * (t - lf), lb[2], lb[3]]
        for t, (b, oc, ot, mt) in enumerate(F):
            kand = [k for k, (lb, lf, lc, vv) in akt.items() if t - lf <= GAP]
            M = np.array([[iou(pred(k, t), b[j]) for j in range(len(b))] for k in kand]) if kand and len(b) else np.zeros((len(kand), len(b)))
            for j in range(len(b)):                       # ambiguous: two tracks overlap the same box well
                if M.shape[0] and (M[:, j] >= 0.3).sum() >= 2: dv += 1; DV.append((ime, t + 1))
            dodj = {}
            if M.size:
                r, c = linear_sum_assignment(-M)
                for i, j in zip(r, c):
                    k = kand[i]; prag = IOU_LINK if t - akt[k][1] == 1 else IOU_GAP
                    if M[i, j] >= prag:
                        dodj[j] = k
                        if t - akt[k][1] > 1: prem += 1
            for j in range(len(b)):
                if j in dodj: k = dodj[j]
                else: nid += 1; k = nid
                c = int(oc[j]) if oc[j] is not None else 0
                vv = (0.0, 0.0)
                if k in akt:
                    lb, lf, lc, v0 = akt[k]; dt = max(1, t - lf)
                    if lc != c: kp += 1
                    vn = ((b[j][0] - lb[0]) / dt, (b[j][1] - lb[1]) / dt)
                    vv = (0.5 * v0[0] + 0.5 * vn[0], 0.5 * v0[1] + 0.5 * vn[1]) if duz[k] > 1 else vn
                akt[k] = (b[j], t, c, vv); duz[k] += 1
                rows.append((t + 1, k, b[j], c))
        ometac = sum(1 for r in rows if KLASE.get(r[3], '').lower() not in PLOVILA)
        os.makedirs(OD + '/%s/gt' % ime, exist_ok=True)
        with open(OD + '/%s/gt/gt.txt' % ime, 'w') as f:
            for (fr, k, bb, c) in rows:
                f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,%d,1\n' % (fr, k, bb[0], bb[1], bb[2], bb[3], c))
        P('  %-18s %6d %7d %5d %8.0f %7d %6d %7d %6d' % (ime[:18], len(F), len(rows), nid, np.mean(list(duz.values())) if duz else 0, dv, prem, kp, ometac))
        UK.update(frm=len(F), okv=len(rows), id=nid, dv=dv, prem=prem, kp=kp, om=ometac)
    P('  TOTAL: %d frames (%.1f min at 30 fps), %d boxes (distractors %d), %d derived IDs; ambiguous boxes %d (%.2f %%), bridged gaps %d, class changes within a track %d' % (
        UK['frm'], UK['frm'] / 30 / 60, UK['okv'], UK['om'], UK['id'], UK['dv'], 100.0 * UK['dv'] / max(1, UK['okv']), UK['prem'], UK['kp']))
    P('  SMD-Plus classes: ' + ', '.join('%d = %s%s' % (k, v, '' if v.lower() in PLOVILA else ' (DISTRACTOR)') for k, v in sorted(KLASE.items())))
    json.dump({str(k): v for k, v in KLASE.items()}, open(OD + '/klase.json', 'w'))
    if DV:
        po = collections.Counter(i for i, _ in DV)
        P('  ambiguous places per sequence (most): ' + ', '.join('%s %d' % kv for kv in po.most_common(6)))
        P('  examples (sequence, frame): ' + ', '.join('%s %d' % x for x in DV[:: max(1, len(DV) // 8)][:8]))
    spremi()

    # ========================================================================= B. check videos
    P(); P('=== B. CHECK VIDEOS WITH DERIVED IDs (check by eye: does the same vessel keep the same number)')
    if cv2 is None: P('  skipped: no cv2')
    else:
        os.makedirs(OD + '/kontrola', exist_ok=True)
        boje = [(int(h * 47 % 255), int(h * 91 % 255), int(h * 151 % 255)) for h in range(1, 400)]
        for ime, m, v in SN:
            if ime not in KONTROLA: continue
            G = ucitaj_gt_mot(OD + '/%s/gt/gt.txt' % ime)
            cap = cv2.VideoCapture(v); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            p = OD + '/kontrola/%s_ID.mp4' % ime
            vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'mp4v'), fps, (960, 540)); t = 0
            while True:
                ok, img = cap.read()
                if not ok: break
                t += 1
                for k, bb, c in G.get(t, []):
                    x, y, w, h = [int(q) for q in bb]; col = boje[k % len(boje)]
                    om = KLASE.get(c, '').lower() not in PLOVILA
                    cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 128) if om else col, 3)
                    cv2.putText(img, ('o%d' if om else '%d') % k, (x, max(30, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 5, cv2.LINE_AA)
                    cv2.putText(img, ('o%d' if om else '%d') % k, (x, max(30, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (128, 128, 128) if om else col, 2, cv2.LINE_AA)
                cv2.putText(img, '%s  frame %d  (number = derived ID; o = distractor)' % (ime, t), (20, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
                vw.write(cv2.resize(img, (960, 540)))
            cap.release(); vw.release(); P('  saved %s (%d frames)' % (p, t))
    spremi()

    # ========================================================================= C. re-encoding
    P(); P('=== C. RE-ENCODING THE VIDEOS TO H.264 (libx264, crf 18, same number of frames)')
    lose = []
    for ime, m, v in SN:
        dst = OD + '/%s/%s.mp4' % (ime, ime)
        nGT = max(ucitaj_gt_mot(OD + '/%s/gt/gt.txt' % ime) or {0: 0})
        if not (os.path.exists(dst) and cv2 is not None and int(cv2.VideoCapture(dst).get(cv2.CAP_PROP_FRAME_COUNT)) >= nGT):
            r = subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', v, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast',
                                '-crf', '18', '-an', '-vsync', 'passthrough', dst], capture_output=True, text=True)
            if r.returncode != 0: lose.append(ime); P('  !! %s: %s' % (ime, r.stderr[-200:])); continue
        if cv2 is not None:
            c = cv2.VideoCapture(dst); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
            if n < nGT: lose.append(ime); P('  !! %s: video %d frames < GT %d' % (ime, n, nGT))
    P('  re-encoded %d/%d%s' % (len(SN) - len(lose), len(SN), ('; FAILED: ' + ', '.join(lose)) if lose else ''))
    spremi()
else:
    KLASE = {int(k): v for k, v in json.load(open(OD + '/klase.json')).items()}

# ============================================================================= D. detector
P(); P('=== D. QUICK DETECTOR CHECK (every 30th frame, threshold 0.25, IoU >= 0.5; a box on a distractor is not counted)')
used = 0
try: used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
except Exception: pass
mod = {}
if used > 1500:
    P('  skipped: GPU is busy (%d MiB) -- run later with --samo-detektor' % used)
else:
    try:
        from ultralytics import YOLO
        os.chdir(W)
        if os.path.exists(W + '/det_glavni/B_yolo11m/weights/best.pt'): mod['SSMOT'] = YOLO(W + '/det_glavni/B_yolo11m/weights/best.pt')
        mod['COCO'] = YOLO('yolo11m.pt'); BOAT = [k for k, n in mod['COCO'].names.items() if n == 'boat']
    except Exception as ex:
        P('  skipped: %s' % str(ex)[:150])
if mod:
    BINS = [(0, 20), (20, 40), (40, 80), (80, 160), (160, 1e9)]; BT = ['<20', '20-40', '40-80', '80-160', '>=160']
    def kos(h): return next(i for i, (a, b) in enumerate(BINS) if a <= h < b)
    CFG = [(m, z) for m in ('SSMOT', 'COCO') if m in mod for z in (640, 1280)]
    hit = {c: [0] * 5 for c in CFG}; ntp = {c: 0 for c in CFG}; nd = {c: 0 for c in CFG}; nneu = {c: 0 for c in CFG}
    ngt = [0] * 5; ms = {c: 0.0 for c in CFG}; nf = 0; t0 = time.time()
    for ime, m, v in SN:
        G = ucitaj_gt_mot(OD + '/%s/gt/gt.txt' % ime)
        cap = cv2.VideoCapture(v); t = 0
        while True:
            ok = cap.grab()
            if not ok: break
            t += 1
            if (t - 1) % 30: continue
            ok, img = cap.retrieve()
            if not ok: break
            nf += 1
            g = G.get(t, []); ev = [bb for k, bb, c in g if KLASE.get(c, '').lower() in PLOVILA]; om = [bb for k, bb, c in g if KLASE.get(c, '').lower() not in PLOVILA]
            for bb in ev: ngt[kos(bb[3])] += 1
            for c in CFG:
                kw = dict(imgsz=c[1], conf=0.25, iou=0.45, verbose=False, half=False)
                if c[0] == 'COCO': kw['classes'] = BOAT
                ta = time.time(); r = mod[c[0]].predict(img, **kw)[0]; ms[c] += time.time() - ta
                D = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy().tolist()]; nd[c] += len(D)
                par = sorted(((iou(a, d), i, j) for i, a in enumerate(ev) for j, d in enumerate(D)), reverse=True); ug, ud = set(), set()
                for u, i, j in par:
                    if u < 0.5: break
                    if i in ug or j in ud: continue
                    ug.add(i); ud.add(j); hit[c][kos(ev[i][3])] += 1; ntp[c] += 1
                nneu[c] += sum(1 for j, d in enumerate(D) if j not in ud and any(iou(o, d) >= 0.5 for o in om))
            if nf % 50 == 0: print('  %d frames (%.0f s)' % (nf, time.time() - t0), flush=True)
        cap.release()
    N = sum(ngt)
    P('  frames %d, vessel boxes %d (by height in px at 1080: %s)' % (nf, N, ', '.join('%s %d' % x for x in zip(BT, ngt))))
    P('  %-11s %s  %8s %10s %8s' % ('detector', ' '.join('%7s' % b for b in BT), 'recall', 'precision', 'ms/frm'))
    for c in CFG:
        P('  %-11s %s  %7.1f%% %9.1f%% %8.0f' % ('%s %d' % c, ' '.join('%6.1f%%' % (100.0 * h / max(1, n)) for h, n in zip(hit[c], ngt)),
          100.0 * ntp[c] / max(1, N), 100.0 * ntp[c] / max(1, nd[c] - nneu[c]), 1000 * ms[c] / max(1, nf)))
P(); P('Done. %s' % (OUT if not SAMO_D else OUT.replace('.txt', '_detektor.txt')))
spremi()
