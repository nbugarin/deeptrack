#!/usr/bin/env python3
"""p71b -- ShipsMOT: fixed or moving camera, and video length against the annotation (read only).

Rule set BEFORE the measurement and before any tracker was run: the background of every 15th frame (0.5 s at 30 fps)
is reduced to a width of 640 px, the annotated vessel boxes (enlarged by 10 %) are masked, and the shift between
consecutive sampled frames is measured by phase correlation. A sequence is FIXED if the median shift is < 1 px and the
95th percentile < 3 px; otherwise it is MOVING. The fixed sequences are the main sample of p73.
Result in the article: 14 fixed sequences (1, 13, 18, 34, 35, 37, 40, 55, 58, 59, 62, 71, 92, 97) and 26 moving.

Also prints the descriptive text files of the dataset and compares the number of frames of NNN.mp4 with gt.txt.

Run:     python3 p71b_shipsmot_kamera.py /path/to/test_shipmot
Output:  screen + p71b_shipsmot_kamera.txt next to the script
"""
import os, sys, glob, cv2, numpy as np, collections
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/test_shipmot"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p71b_shipsmot_kamera.txt")
L = []
def say(s=""): print(s); L.append(str(s))
seqs = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))], key=lambda x: int(x) if x.isdigit() else 0)

say("=== Structure of one sequence (%s) ===" % seqs[0])
for dp, dn, fn in os.walk(os.path.join(ROOT, seqs[0])):
    say(f"  {os.path.relpath(dp, ROOT)}/  folders={sorted(dn)}  files={len(fn)}  example={sorted(fn)[:3]}")
say("\n=== Descriptive files ===")
for f in sorted(glob.glob(os.path.join(ROOT, "*", "*.txt")) + glob.glob(os.path.join(ROOT, "**", "*.cfg"), recursive=True)):
    say(f"--- {os.path.relpath(f, ROOT)}")
    say(open(f, errors="replace").read()[:600])

say("\n=== mp4 vs gt and camera shift (step 15 frames = 0.5 s, image 640 px) ===")
say("seq | gtFr | mp4Fr | medShift px | p95 px | maxShift | decision")
dec = collections.Counter()
for s in seqs:
    sd = os.path.join(ROOT, s)
    gt = glob.glob(os.path.join(sd, "**", "gt.txt"), recursive=True)
    if not gt: continue
    boxes = collections.defaultdict(list)
    for ln in open(gt[0]):
        p = [float(x) for x in ln.replace(",", " ").split()]
        boxes[int(p[0])].append(p[2:6])
    nfr = max(boxes)
    mp4 = glob.glob(os.path.join(sd, "**", "*.mp4"), recursive=True)
    mfr = int(cv2.VideoCapture(mp4[0]).get(cv2.CAP_PROP_FRAME_COUNT)) if mp4 else -1
    imgs = sorted(f for f in glob.glob(os.path.join(sd, "**", "*.jpg"), recursive=True))
    sh, prev = [], None
    for k in range(0, len(imgs), 15):
        im = cv2.imread(imgs[k], cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        H, W = im.shape; sc = 640.0 / W
        g = cv2.resize(im, (640, int(H * sc))).astype(np.float32)
        m = np.ones_like(g)
        for x, y, w, h in boxes.get(k + 1, []):
            x0, y0 = max(0, int((x - 0.1*w) * sc)), max(0, int((y - 0.1*h) * sc))
            m[y0:int((y + 1.1*h) * sc), x0:int((x + 1.1*w) * sc)] = 0
        g = g * m + (1 - m) * (g[m > 0].mean() if (m > 0).any() else 0)
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g, cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F))
            sh.append(float(np.hypot(dx, dy)))
        prev = g
    if not sh: continue
    med, p95, mx = np.median(sh), np.percentile(sh, 95), max(sh)
    d = "fixed" if med < 1 and p95 < 3 else "moving"; dec[d] += 1
    say(f"{s} | {nfr} | {mfr} | {med:.2f} | {p95:.2f} | {mx:.1f} | {d}")
say(f"\nSummary: {dict(dec)}")
open(OUT, "w").write("\n".join(L) + "\n"); print(f"\nWritten: {OUT}")
