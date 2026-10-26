#!/usr/bin/env python3
"""p60e -- BRZA PROVJERA DETEKTORA NA FVESSELU (ultralytics, bez DeepStreama i bez trackera).

U p60_dijagnoza dio C je dao 0 detekcija na svim ulazima, iako isti model u DeepStreamu nalazi brodove -> sumnja na
FP16 (half) u PyTorchu na ovoj grafickoj. Ovdje:
 1) tri frejma u detalje: SSMOT YOLO11m (best.pt) i opci COCO YOLO11m (samo klasa 'boat'), ulaz 640 / 1280, FP32 i FP16
 2) odziv na svakoj 10. sekundi s GT-om (sve snimke), FP32: SSMOT 640 / 1280 i COCO 640 / 1280

COCO yolo11m.pt se jednom skine (~40 MB) u /workspace. Nista se ne mijenja u postojecim datotekama.
Pokretanje:  python3 /workspace/p60_detektor.py /workspace/Clip-10_h264
Izlaz:       /workspace/scpilot/p60_detektor.txt   (dio 1 je gotov za ~1-2 min, dio 2 za ~5-10 min)
"""
import os, sys, glob, time
import numpy as np
import cv2

W = os.environ.get('P60_W', '/workspace')
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
FV = ARGS[0] if ARGS else W + '/Clip-10_h264'
OUT = W + '/scpilot/p60_detektor.txt'
PT = W + '/det_glavni/B_yolo11m/weights/best.pt'
K_GLOB = 14; CONF = 0.25; NMS = 0.45; KORAK = 10
BINS = [(0, 30), (30, 50), (50, 80), (80, 150), (150, 1e9)]
BIN_TXT = ['<30', '30-50', '50-80', '80-150', '>=150']
os.makedirs(os.path.dirname(OUT), exist_ok=True)
L = []
def P(x=''):
    print(x, flush=True); L.append(str(x)); open(OUT, 'w').write('\n'.join(L) + '\n')

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
def kos(h):
    for i, (a, b) in enumerate(BINS):
        if a <= h < b: return i
    return len(BINS) - 1
def pc(a, b): return ('%.1f %%' % (100.0 * a / b)) if b else '-'

SN = []
for d in sorted(glob.glob(os.path.join(FV, 'clip-*'))):
    v = glob.glob(d + '/*.mp4'); g = glob.glob(d + '/gt/*_gt_tracking.txt')
    if len(v) == 1 and len(g) == 1: SN.append((os.path.basename(d), v[0], ucitaj_gt(g[0])))
if not SN: sys.exit('u %s nema snimki clip-XX s videom i gt/*_gt_tracking.txt' % FV)

try:
    import torch, ultralytics
    from ultralytics import YOLO
except Exception as ex:
    sys.exit('nema torch/ultralytics: %s' % ex)
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NEMA CUDA-e (radi na CPU-u)'
P('p60e -- torch %s, ultralytics %s, graficka: %s' % (torch.__version__, ultralytics.__version__, gpu))
os.chdir(W)
MOD = {}
try:
    MOD['SSMOT'] = YOLO(PT); P('SSMOT model: %s, klase %s' % (PT, list(MOD['SSMOT'].names.values())))
except Exception as ex:
    P('SSMOT model nije ucitan: %s' % str(ex)[:150])
try:
    MOD['COCO'] = YOLO('yolo11m.pt')
    BOAT = [k for k, n in MOD['COCO'].names.items() if n == 'boat']
    P('COCO model: yolo11m.pt, koristi se samo klasa boat %s' % BOAT)
except Exception as ex:
    P('COCO model nije ucitan (nema mreze?): %s' % str(ex)[:150])

def detekcije(ime, img, z, half):
    m = MOD[ime]; kw = dict(imgsz=z, conf=CONF, iou=NMS, verbose=False, half=half)
    if ime == 'COCO': kw['classes'] = BOAT
    ta = time.time(); r = m.predict(img, **kw)[0]; ms = 1000 * (time.time() - ta)
    xyxy = r.boxes.xyxy.cpu().numpy().tolist(); cls = r.boxes.cls.cpu().numpy().tolist(); cf = r.boxes.conf.cpu().numpy().tolist()
    return [([x1, y1, x2 - x1, y2 - y1], m.names[int(k)], float(c)) for (x1, y1, x2, y2), k, c in zip(xyxy, cls, cf)], ms

# ============================================================================= 1. tri frejma u detalje
P(); P('=== 1. TRI FREJMA U DETALJE (prag %.2f; za svaki GT: najbolji IoU, pouzdanost i klasa tog okvira)' % CONF)
CFG1 = [(m, z, h) for m in ('SSMOT', 'COCO') if m in MOD for z in (640, 1280) for h in (False, True)]
for ime, v, G in SN:
    if ime not in ('clip-01', 'clip-05', 'clip-08'): continue
    sek = sorted(G); s = 2 if ime == 'clip-01' and 2 in G else sek[len(sek) // 2]
    cap = cv2.VideoCapture(v); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; f = int(round(s * fps)) + K_GLOB
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read(); cap.release()
    if not ok: P('  %s frejm %d nije procitan' % (ime, f)); continue
    Wk, Hk = img.shape[1], img.shape[0]
    P('  %s sekunda %d (frejm %d): %s' % (ime, s, f, '  '.join('GT%d [%d,%d,%d,%d]' % ((g,) + tuple(int(q) for q in b)) for g, b in G[s])))
    for (m, z, h) in CFG1:
        D, ms = detekcije(m, img, z, h)
        opis = []
        for g, b in G[s]:
            best = max(D, key=lambda d: iou(b, d[0])) if D else None
            opis.append('GT%d %s' % (g, ('IoU %.2f (%.2f %s)' % (iou(b, best[0]), best[2], best[1][:12])) if best and iou(b, best[0]) > 0 else 'nista'))
        ndiv = sum(1 for d in D if d[0][2] * d[0][3] > 0.10 * Wk * Hk)
        P('    %-5s %4d %s  okvira %2d (divovskih %d)  %4.0f ms | %s' % (m, z, 'FP16' if h else 'FP32', len(D), ndiv, ms, ' | '.join(opis)))

# ============================================================================= 2. odziv na svakoj 10. sekundi
CFG2 = [(m, z) for m in ('SSMOT', 'COCO') if m in MOD for z in (640, 1280)]
P(); P('=== 2. ODZIV NA SVAKOJ %d. SEKUNDI S GT-om (sve snimke, FP32, prag %.2f, IoU >= 0,5 jedan-na-jedan)' % (KORAK, CONF))
hit = {c: [0] * len(BINS) for c in CFG2}; ndet = {c: 0 for c in CFG2}; ntp = {c: 0 for c in CFG2}; ndiv = {c: 0 for c in CFG2}
vr = {c: 0.0 for c in CFG2}; ngt = [0] * len(BINS); nfrm = 0; po = {c: [] for c in CFG2}; t0 = time.time()
for ime, v, G in SN:
    cap = cv2.VideoCapture(v); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9
    treba = {}
    for s in G:
        if s % KORAK == 0:
            f = int(round(s * fps)) + K_GLOB
            if 0 <= f < nfr: treba[f] = s
    lok = {c: [0, 0] for c in CFG2}; f = 0; zadnji = max(treba) if treba else -1
    while f <= zadnji:
        if f in treba:
            ok, img = cap.read()
            if not ok: break
            gl = [b for _, b in G[treba[f]]]; nfrm += 1; Wk, Hk = img.shape[1], img.shape[0]
            for b in gl: ngt[kos(b[3])] += 1
            for c in CFG2:
                D, ms = detekcije(c[0], img, c[1], False); vr[c] += ms
                dets = [d[0] for d in D]; ndet[c] += len(dets); ndiv[c] += sum(1 for b in dets if b[2] * b[3] > 0.10 * Wk * Hk)
                par = sorted(((iou(g, d), gi, di) for gi, g in enumerate(gl) for di, d in enumerate(dets)), reverse=True)
                ug, ud = set(), set()
                for u, gi, di in par:
                    if u < 0.5: break
                    if gi in ug or di in ud: continue
                    ug.add(gi); ud.add(di); hit[c][kos(gl[gi][3])] += 1; ntp[c] += 1
                lok[c][0] += len(ug); lok[c][1] += len(gl)
        else:
            if not cap.grab(): break
        f += 1
    cap.release()
    for c in CFG2: po[c].append((ime, lok[c][0], lok[c][1]))
    print('  %s gotovo (%.0f s)' % (ime, time.time() - t0), flush=True)
N = sum(ngt)
P('  frejmova %d, GT okvira %d (po visini: %s)' % (nfrm, N, ', '.join('%s %d' % (t, n) for t, n in zip(BIN_TXT, ngt))))
P('  %-12s %s   %8s  %10s  %15s  %8s' % ('detektor', '  '.join('%7s' % t for t in BIN_TXT), 'ukupno', 'preciznost', 'divovskih/frejm', 'ms/frejm'))
for c in CFG2:
    P('  %-12s %s   %8s  %10s  %15.2f  %8.0f' % ('%s %d' % c, '  '.join('%7s' % pc(h, m) for h, m in zip(hit[c], ngt)), pc(ntp[c], N),
      pc(ntp[c], ndet[c]), ndiv[c] / max(1, nfrm), vr[c] / max(1, nfrm)))
P('  po snimci (odziv, redom %s):' % ', '.join('%s %d' % c for c in CFG2))
for j, (ime, _, _) in enumerate(SN):
    P('    %s  ' % ime + '  '.join('%7s' % pc(po[c][j][1], po[c][j][2]) for c in CFG2))
P('  (usporedba: DeepStream + tracker vlastiti pokriva 8,2 % GT-a u prozoru +-1-2 s, 4,2 % na tocnom frejmu)')
P(); P('Gotovo. %s' % OUT)
