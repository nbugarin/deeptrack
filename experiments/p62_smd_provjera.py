#!/usr/bin/env python3
"""p62 -- CHECK OF SMD-Plus (Singapore Maritime Dataset Plus): annotation format, whether track IDs exist, videos.

Read only; runs nothing on the GPU and changes nothing.
 1) list of videos and .mat files (ObjectGT), matching by name, resolution / fps / number of frames
 2) structure of one .mat file: variables, fields, first few frames -- is there a field with an ID
 3) for all clips: annotated frames, objects per frame, classes
 4) if there is no ID: can an ID be derived automatically -- is the same object index in neighbouring frames the same vessel
    (order), how many pairs are unambiguous (IoU >= 0.5, second candidate < 0.3), how many ambiguous

Run:     python3 /workspace/p62_smd_provjera.py /workspace/SMD_Plus/VIS_Onshore
         (give the folder that contains the videos and ObjectGT; subfolders are searched too)
Output:  /workspace/scpilot/p62_smd_provjera.txt
"""
import os, sys, glob, re, collections
import numpy as np

W = os.environ.get('P60_W', '/workspace')
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
ROOT = ARGS[0] if ARGS else W + '/SMD_Plus/VIS_Onshore'
OUT = W + '/scpilot/p62_smd_provjera.txt'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
L = []
def P(x=''):
    print(x, flush=True); L.append(str(x)); open(OUT, 'w').write('\n'.join(L) + '\n')
try:
    import cv2
except ImportError:
    cv2 = None

if not os.path.isdir(ROOT): sys.exit('no folder %s -- give the path to the extracted folder (e.g. .../VIS_Onshore)' % ROOT)
vid = sorted(p for e in ('avi', 'mp4', 'mov', 'AVI', 'MP4', 'MOV') for p in glob.glob(ROOT + '/**/*.' + e, recursive=True))
mats = sorted(glob.glob(ROOT + '/**/*.mat', recursive=True))
P('p62 SMD-Plus check -- %s' % ROOT)
P('videos: %d, .mat files: %d' % (len(vid), len(mats)))
for p in mats[:5]: P('  example .mat: %s' % os.path.relpath(p, ROOT))
for p in vid[:3]: P('  example video: %s' % os.path.relpath(p, ROOT))
if not mats: sys.exit('no .mat files')

def kljuc(p):                                            # MVI_1448_VIS_ObjectGT.mat -> MVI_1448_VIS
    b = os.path.splitext(os.path.basename(p))[0]
    return re.sub(r'[_-]?ObjectGT$', '', b, flags=re.I)
VK = {kljuc(v): v for v in vid}
PAR = [(kljuc(m), m, VK.get(kljuc(m))) for m in mats]
bez = [k for k, _, v in PAR if v is None]
P('matched .mat + video: %d%s' % (len(PAR) - len(bez), ('; without video: ' + ', '.join(bez[:8])) if bez else ''))

# ----------------------------------------------------------------------------- loading .mat (v5 or v7.3)
def ucitaj_mat(p):
    try:
        import scipy.io as sio
        d = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
        return 'v5', {k: v for k, v in d.items() if not k.startswith('__')}
    except NotImplementedError:
        import h5py
        return 'v7.3', h5py.File(p, 'r')
def polja(o):
    return list(getattr(o, '_fieldnames', []) or [])
def opis(v, n=200):
    if isinstance(v, np.ndarray): return 'ndarray%s %s: %s' % (v.shape, v.dtype, np.array2string(v.ravel()[:12], max_line_width=n))
    return '%s: %s' % (type(v).__name__, str(v)[:n])

# ----------------------------------------------------------------------------- 2. structure of the first file
P(); P('=== 2. STRUCTURE: %s' % os.path.basename(PAR[0][1]))
ver, D = ucitaj_mat(PAR[0][1])
P('format MATLAB %s' % ver)
if ver == 'v7.3':
    def obidi(g, pref=''):
        for k in g:
            o = g[k]; P('  %s%s: %s' % (pref, k, ('group' if not hasattr(o, 'shape') else 'set %s %s' % (o.shape, o.dtype))))
            if not hasattr(o, 'shape'): obidi(o, pref + k + '/')
    obidi(D); P('!! v7.3 format -- report this output (reading must be adapted)'); sys.exit(0)
GLAVNA = None
for k, v in D.items():
    P('  variable %s: %s' % (k, type(v).__name__ if not isinstance(v, np.ndarray) else 'ndarray%s %s' % (v.shape, v.dtype)))
    el = v.ravel()[0] if isinstance(v, np.ndarray) and v.dtype == object and v.size else v
    if polja(el):
        P('    fields: %s' % polja(el))
        if isinstance(v, np.ndarray) and (GLAVNA is None or v.size > D[GLAVNA].size): GLAVNA = k
if GLAVNA is None: P('!! no array of per-frame structures found -- report this output'); sys.exit(0)
S = D[GLAVNA].ravel(); F = polja(S[0])
P('main variable: %s (%d elements = frames?), fields %s' % (GLAVNA, S.size, F))
for i in range(min(3, S.size)):
    P('  element %d:' % i)
    for f in F: P('    %-10s %s' % (f, opis(getattr(S[i], f))))
IDP = [f for f in F if re.search(r'id|track|trag|label_?id', f, re.I)]
BBP = [f for f in F if re.search(r'^bb$|bbox|box', f, re.I)]
KLP = [f for f in F if re.search(r'object|class|cls', f, re.I)]
P('field with ID: %s | boxes: %s | class: %s' % (IDP or 'NONE', BBP or '?', KLP or '?'))
if not BBP: P('!! field with boxes not recognised -- report this output'); sys.exit(0)

def okviri(e):
    b = np.asarray(getattr(e, BBP[0]), dtype=float)
    if b.size == 0: return np.zeros((0, 4))
    return np.atleast_2d(b).reshape(-1, 4)
def klase(e, n):
    if not KLP: return [None] * n
    c = getattr(e, KLP[0]); c = np.atleast_1d(np.asarray(c)).ravel().tolist()
    return (c + [None] * n)[:n]
def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[0] + a[2], b[0] + b[2]); y2 = min(a[1] + a[3], b[1] + b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1); u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0

# ----------------------------------------------------------------------------- 3. + 4. all clips
P(); P('=== 3./4. ALL CLIPS (video: resolution, fps, frames | GT: frames, objects/frame | ID from order | unambiguous pairs)')
P('  %-18s %-24s %7s %6s %8s %9s %9s' % ('sequence', 'video', 'GT frm', 'obj/f', 'max obj', 'order', 'unamb.'))
KL = collections.Counter(); uk = collections.Counter(); xmax = 0; ymax = 0
for k, m, v in PAR:
    try:
        ver, D = ucitaj_mat(m); S = D[GLAVNA].ravel()
    except Exception as ex:
        P('  %-18s !! not loaded: %s' % (k, str(ex)[:80])); continue
    vo = '-'
    if v and cv2 is not None:
        c = cv2.VideoCapture(v); vo = '%dx%d %.1ffps %d' % (c.get(3), c.get(4), c.get(cv2.CAP_PROP_FPS), c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
    B = [okviri(e) for e in S]; nobj = [len(b) for b in B]
    for e, b in zip(S, B):
        KL.update(str(x) for x in klase(e, len(b)))
        if len(b): xmax = max(xmax, float((b[:, 0] + b[:, 2]).max())); ymax = max(ymax, float((b[:, 1] + b[:, 3]).max()))
    red = [0, 0]; jed = [0, 0]
    for a, b in zip(B[:-1], B[1:]):
        if len(a) == 0 or len(b) == 0: continue
        for i in range(len(a)):
            s = sorted(((iou(a[i], b[j]), j) for j in range(len(b))), reverse=True)
            if s[0][0] < 0.3: continue
            red[1] += 1; red[0] += int(s[0][1] == i)
            jed[1] += 1; jed[0] += int(s[0][0] >= 0.5 and (len(s) < 2 or s[1][0] < 0.3))
    uk.update(frm=len(S), obj=sum(nobj), red0=red[0], red1=red[1], jed0=jed[0], jed1=jed[1])
    P('  %-18s %-24s %7d %6.1f %8d %8.1f%% %8.1f%%' % (k[:18], vo, len(S), np.mean(nobj) if nobj else 0, max(nobj) if nobj else 0,
      100.0 * red[0] / max(1, red[1]), 100.0 * jed[0] / max(1, jed[1])))
P()
P('TOTAL: %d clips, %d annotated frames, %d boxes; largest x+w %.0f, y+h %.0f' % (len(PAR), uk['frm'], uk['obj'], xmax, ymax))
P('classes: ' + ', '.join('%s: %d' % (c, n) for c, n in KL.most_common()))
P('same object index = same vessel in the neighbouring frame: %.1f %% of pairs' % (100.0 * uk['red0'] / max(1, uk['red1'])))
P('unambiguous linking by overlap (IoU >= 0.5, second < 0.3): %.1f %% of pairs' % (100.0 * uk['jed0'] / max(1, uk['jed1'])))
P()
P('Interpretation: if there is a field with an ID -> the set is ready for tracking evaluation. If not, and the order is ~100 %,')
P('the ID is in fact the object index. If not even that, the ID is derived by linking by overlap (with the stated limitation).')
P('Result: %s' % OUT)
