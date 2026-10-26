#!/usr/bin/env python3
"""p62 -- PROVJERA SMD-Plus (Singapore Maritime Dataset Plus): format anotacija, ima li ID-ova tragova, videi.

Samo cita; nista ne pokrece na GPU-u i nista ne mijenja.
 1) popis videa i .mat datoteka (ObjectGT), uparivanje po imenu, rezolucija / fps / broj frejmova
 2) struktura jedne .mat datoteke: varijable, polja, prvih nekoliko frejmova -- ima li polje s ID-om
 3) za sve snimke: frejmova s anotacijom, objekata po frejmu, klase
 4) ako nema ID-a: moze li se ID izvesti automatski -- je li isti redni broj objekta u susjednim frejmovima isti brod
    (redoslijed), koliko je parova jednoznacno (IoU >= 0,5, drugi kandidat < 0,3), koliko dvosmislenih

Pokretanje:  python3 /workspace/p62_smd_provjera.py /workspace/SMD_Plus/VIS_Onshore
             (zadaj mapu koja sadrzi videe i ObjectGT; trazi i u podmapama)
Izlaz:       /workspace/scpilot/p62_smd_provjera.txt
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

if not os.path.isdir(ROOT): sys.exit('nema mape %s -- zadaj put do raspakirane mape (npr. .../VIS_Onshore)' % ROOT)
vid = sorted(p for e in ('avi', 'mp4', 'mov', 'AVI', 'MP4', 'MOV') for p in glob.glob(ROOT + '/**/*.' + e, recursive=True))
mats = sorted(glob.glob(ROOT + '/**/*.mat', recursive=True))
P('p62 SMD-Plus provjera -- %s' % ROOT)
P('videa: %d, .mat datoteka: %d' % (len(vid), len(mats)))
for p in mats[:5]: P('  primjer .mat: %s' % os.path.relpath(p, ROOT))
for p in vid[:3]: P('  primjer videa: %s' % os.path.relpath(p, ROOT))
if not mats: sys.exit('nema .mat datoteka')

def kljuc(p):                                            # MVI_1448_VIS_ObjectGT.mat -> MVI_1448_VIS
    b = os.path.splitext(os.path.basename(p))[0]
    return re.sub(r'[_-]?ObjectGT$', '', b, flags=re.I)
VK = {kljuc(v): v for v in vid}
PAR = [(kljuc(m), m, VK.get(kljuc(m))) for m in mats]
bez = [k for k, _, v in PAR if v is None]
P('uparenih .mat + video: %d%s' % (len(PAR) - len(bez), ('; bez videa: ' + ', '.join(bez[:8])) if bez else ''))

# ----------------------------------------------------------------------------- ucitavanje .mat (v5 ili v7.3)
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

# ----------------------------------------------------------------------------- 2. struktura prve datoteke
P(); P('=== 2. STRUKTURA: %s' % os.path.basename(PAR[0][1]))
ver, D = ucitaj_mat(PAR[0][1])
P('format MATLAB %s' % ver)
if ver == 'v7.3':
    def obidi(g, pref=''):
        for k in g:
            o = g[k]; P('  %s%s: %s' % (pref, k, ('grupa' if not hasattr(o, 'shape') else 'skup %s %s' % (o.shape, o.dtype))))
            if not hasattr(o, 'shape'): obidi(o, pref + k + '/')
    obidi(D); P('!! v7.3 format -- posalji ovaj ispis Claudeu (treba prilagoditi citanje)'); sys.exit(0)
GLAVNA = None
for k, v in D.items():
    P('  varijabla %s: %s' % (k, type(v).__name__ if not isinstance(v, np.ndarray) else 'ndarray%s %s' % (v.shape, v.dtype)))
    el = v.ravel()[0] if isinstance(v, np.ndarray) and v.dtype == object and v.size else v
    if polja(el):
        P('    polja: %s' % polja(el))
        if isinstance(v, np.ndarray) and (GLAVNA is None or v.size > D[GLAVNA].size): GLAVNA = k
if GLAVNA is None: P('!! nisam nasao niz struktura po frejmovima -- posalji ovaj ispis Claudeu'); sys.exit(0)
S = D[GLAVNA].ravel(); F = polja(S[0])
P('glavna varijabla: %s (%d elemenata = frejmova?), polja %s' % (GLAVNA, S.size, F))
for i in range(min(3, S.size)):
    P('  element %d:' % i)
    for f in F: P('    %-10s %s' % (f, opis(getattr(S[i], f))))
IDP = [f for f in F if re.search(r'id|track|trag|label_?id', f, re.I)]
BBP = [f for f in F if re.search(r'^bb$|bbox|box', f, re.I)]
KLP = [f for f in F if re.search(r'object|class|cls', f, re.I)]
P('polje s ID-om: %s | okviri: %s | klasa: %s' % (IDP or 'NEMA', BBP or '?', KLP or '?'))
if not BBP: P('!! ne prepoznajem polje s okvirima -- posalji ovaj ispis Claudeu'); sys.exit(0)

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

# ----------------------------------------------------------------------------- 3. + 4. sve snimke
P(); P('=== 3./4. SVE SNIMKE (video: rezolucija, fps, frejmova | GT: frejmova, objekata/frejm | ID iz redoslijeda | jednoznacni parovi)')
P('  %-18s %-24s %7s %6s %8s %9s %9s' % ('snimka', 'video', 'GT frm', 'obj/f', 'max obj', 'redosl.', 'jednozn.'))
KL = collections.Counter(); uk = collections.Counter(); xmax = 0; ymax = 0
for k, m, v in PAR:
    try:
        ver, D = ucitaj_mat(m); S = D[GLAVNA].ravel()
    except Exception as ex:
        P('  %-18s !! nije ucitan: %s' % (k, str(ex)[:80])); continue
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
P('UKUPNO: %d snimki, %d anotiranih frejmova, %d okvira; najveci x+w %.0f, y+h %.0f' % (len(PAR), uk['frm'], uk['obj'], xmax, ymax))
P('klase: ' + ', '.join('%s: %d' % (c, n) for c, n in KL.most_common()))
P('isti redni broj objekta = isti brod u susjednom frejmu: %.1f %% parova' % (100.0 * uk['red0'] / max(1, uk['red1'])))
P('jednoznacno povezivanje po preklapanju (IoU >= 0,5, drugi < 0,3): %.1f %% parova' % (100.0 * uk['jed0'] / max(1, uk['jed1'])))
P()
P('Tumacenje: ako postoji polje s ID-om -> skup je spreman za ocjenu praćenja. Ako ga nema, a redoslijed je ~100 %,')
P('ID je zapravo redni broj objekta. Ako ni to, ID se izvodi povezivanjem po preklapanju (uz navedeno ogranicenje).')
P('Rezultat: %s' % OUT)
