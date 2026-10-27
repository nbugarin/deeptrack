#!/usr/bin/env python3
"""p71c -- ShipsMOT: class number in gt.txt -> class name (read only).

gt.txt gives the class as a number; NNN.txt next to it gives, per box, the frame (0-based), the class name, the
identifier and the box. Boxes are joined on frame and rounded coordinates, and the most frequent names per class number
are printed. Result in the article: 0 Container ship, 1 General cargo ship, 2 Bulk cargo carrier, 3 Ferry boat, 4 Tug,
5 Kayak, 6 Oil carrier, 8 Fishing boat, 9 Passenger ship, 10 Sailing ship, 13 Canoe, 14 Speed boat (7, 11 and 12 do not
occur in the test split). All are vessel types, so p73 evaluates class-agnostically without distractors.

Run:  python3 p71c_shipsmot_klase.py /path/to/test_shipmot
"""
import os, sys, collections
R = sys.argv[1] if len(sys.argv) > 1 else "/workspace/test_shipmot"
m = collections.defaultdict(collections.Counter)
for s in os.listdir(R):
    g = os.path.join(R, s, "gt", "gt.txt"); t = os.path.join(R, s, s + ".txt")
    if not (os.path.exists(g) and os.path.exists(t)): continue
    gt = {}
    for ln in open(g):
        p = [float(x) for x in ln.split(",")]
        gt[(int(p[0]), round(p[2]), round(p[3]), round(p[4]), round(p[5]))] = int(p[7])
    for ln in open(t):
        p = ln.strip().split(",")
        if len(p) < 7: continue
        k = (int(p[0]) + 1, round(float(p[3])), round(float(p[4])), round(float(p[5])), round(float(p[6])))
        if k in gt: m[gt[k]][p[1]] += 1
for c in sorted(m): print(c, dict(m[c].most_common(3)))
