#!/usr/bin/env python3
"""p71 -- check of the ShipsMOT test split (read only; nothing is run).

Prints the folder structure, the number of files per extension and, for every sequence with a gt/gt.txt, the frame
rate, resolution and length from seqinfo.ini, the number of images, annotated frames, boxes and identities, the median
track length, gaps in tracks, boxes extending beyond the image and the classes. Used to fix the rules of p73 before
any tracker was run.

Run:     python3 p71_shipsmot_provjera.py /path/to/test_shipmot
Output:  screen + p71_shipsmot_provjera.txt next to the script
"""
import os, sys, glob, configparser, collections, statistics

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/test_shipmot"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p71_shipsmot_provjera.txt")
IMG = (".jpg", ".jpeg", ".png", ".bmp")
VID = (".mp4", ".avi", ".mov", ".mkv")
lines = []
def say(s=""):
    print(s); lines.append(s)

if not os.path.isdir(ROOT):
    sys.exit(f"No folder {ROOT} -- give the path as an argument.")

say(f"=== p71 ShipsMOT check: {ROOT} ===")
ext = collections.Counter(); nfiles = 0
for dp, dn, fn in os.walk(ROOT):
    for f in fn:
        nfiles += 1; ext[os.path.splitext(f)[1].lower()] += 1
say(f"Files in total: {nfiles}")
say("Extensions: " + ", ".join(f"{e or '(none)'} {n}" for e, n in ext.most_common(12)))
say("Top of the tree:")
for d in sorted(os.listdir(ROOT))[:30]:
    p = os.path.join(ROOT, d)
    say(f"  {d}/ ({len(os.listdir(p))} items)" if os.path.isdir(p) else f"  {d}")
for f in sorted(glob.glob(os.path.join(ROOT, "**", "*"), recursive=True)):
    if f.lower().endswith((".md", ".txt", ".json", ".yaml", ".yml")) and "gt" not in os.path.basename(f).lower() \
            and os.path.getsize(f) < 20000 and f.count(os.sep) - ROOT.count(os.sep) <= 2:
        say(f"  [descriptive file] {os.path.relpath(f, ROOT)}")

gts = sorted(glob.glob(os.path.join(ROOT, "**", "gt.txt"), recursive=True))
say(f"\ngt.txt found: {len(gts)}")
if not gts:
    vids = [f for f in glob.glob(os.path.join(ROOT, "**", "*"), recursive=True) if f.lower().endswith(VID)]
    say(f"Videos: {len(vids)}; examples: {[os.path.relpath(v, ROOT) for v in vids[:5]]}")
else:
    tot = collections.Counter(); ncol = collections.Counter(); confv = collections.Counter()
    fps_c = collections.Counter(); res_c = collections.Counter(); splits = collections.Counter()
    say("seq | split | fps | WxH | seqLen | img | annotFr | boxes | IDs | medTrackLen | gaps | outsideImage | classes")
    for g in gts:
        seq = os.path.dirname(os.path.dirname(g)) if os.path.basename(os.path.dirname(g)) == "gt" else os.path.dirname(g)
        name = os.path.relpath(seq, ROOT)
        split = name.split(os.sep)[0] if os.sep in name else "-"; splits[split] += 1
        fps = W = H = L = None
        ini = os.path.join(seq, "seqinfo.ini")
        if os.path.exists(ini):
            c = configparser.ConfigParser(); c.read(ini); s = c[c.sections()[0]]
            fps = s.get("frameRate"); W = s.get("imWidth"); H = s.get("imHeight"); L = s.get("seqLength")
        nimg = sum(1 for f in glob.glob(os.path.join(seq, "**", "*"), recursive=True) if f.lower().endswith(IMG))
        rows = []
        for ln in open(g):
            p = ln.replace(",", " ").split()
            if len(p) >= 6:
                rows.append([float(x) for x in p]); ncol[len(p)] += 1
        frames = {int(r[0]) for r in rows} or {0}
        tracks = collections.defaultdict(list); cls = collections.Counter(); outside = 0
        for r in rows:
            tracks[int(r[1])].append(int(r[0]))
            if len(r) >= 7: confv[r[6]] += 1
            if len(r) >= 8: cls[int(r[7])] += 1; tot[int(r[7])] += 1
            if W and H and (r[2] < -1 or r[3] < -1 or r[2] + r[4] > float(W) + 1 or r[3] + r[5] > float(H) + 1):
                outside += 1
        lens = [len(v) for v in tracks.values()]
        gaps = sum(1 for v in tracks.values() for a, b in zip(sorted(v), sorted(v)[1:]) if b - a > 1)
        fps_c[fps] += 1; res_c[f"{W}x{H}"] += 1
        say(f"{name} | {split} | {fps} | {W}x{H} | {L} | {nimg} | {len(frames)} ({min(frames)}-{max(frames)}) | "
            f"{len(rows)} | {len(tracks)} | {statistics.median(lens) if lens else 0} | {gaps} | {outside} | "
            f"{dict(cls.most_common(6))}")
    say("\n=== Summary ===")
    say(f"Sequences: {len(gts)}; per folder: {dict(splits)}")
    say(f"FPS: {dict(fps_c)}")
    say(f"Resolutions: {dict(res_c)}")
    say(f"Columns in gt.txt: {dict(ncol)}")
    say(f"Values of column 7 (conf/active): {dict(confv.most_common(5))}")
    say(f"Classes (column 8) in total: {dict(sorted(tot.items()))}")

open(OUT, "w").write("\n".join(lines) + "\n")
print(f"\nWritten: {OUT}")
