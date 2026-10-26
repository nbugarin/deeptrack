#!/usr/bin/env python3
"""p60c -- PRIPREMA OPCEG COCO DETEKTORA (YOLO11m, samo klasa 'boat') ZA DEEPSTREAM, za ponovni p60 na FVesselu.

Zasto: p60_detektor.py je pokazao da SSMOT detektor na rijeci Jangce nalazi tek ~6 % brodova (nema teretnih brodova
ni teglenica medu svojih 9 klasa), a opci COCO YOLO11m bez ikakvog treniranja ~48 % (ulaz 640). Trackeri ostaju
zamrznuti; svi dobivaju iste detekcije. Ulaz 640 = isti kao u cijelom IEEE cjevovodu.

Koraci (nista postojece se ne mijenja; svi engine-i se cuvaju sigurnosnom kopijom i provjeravaju):
 1) ONNX iz /workspace/yolo11m.pt preko DeepStream-Yolo skripte za izvoz (u zasebnoj mapi /workspace/coco_det)
 2) novi config /workspace/DeepStream-Yolo/config_infer_coco11m_640.txt (kopija config_infer_11m.txt; mijenja se samo
    model, labele, broj klasa = 80 i dodaje filtar da prolazi samo klasa 8 = boat)
 3) izgradnja TensorRT engine-a na kratkom isjecku (prvi put traje 3-10 min) -> /workspace/coco_det/yolo11m_coco_640_b1_gpu0_fp16.engine
 4) provjera: drugo pokretanje ucitava engine; ispis detekcija na clip-01 i usporedba s GT-om

Pokretanje (GPU slobodan):  python3 /workspace/p60_coco_priprema.py
Nakon toga:                 nohup python3 /workspace/p60_fvessel.py /workspace/Clip-10_h264 --det=coco > /workspace/p60c_out.txt 2>&1 &
"""
import os, re, sys, json, glob, time, shutil, hashlib, subprocess

W = os.environ.get('P60_W', '/workspace')
C = W + '/coco_det'
PT = W + '/yolo11m.pt'
DSY = W + '/DeepStream-Yolo'
INFER_SRC = DSY + '/config_infer_11m.txt'
INFER_NEW = DSY + '/config_infer_coco11m_640.txt'
ENG = C + '/yolo11m_coco_640_b1_gpu0_fp16.engine'
ONNX = C + '/yolo11m_coco_640.onnx'
LAB = C + '/labels_coco.txt'
PTJ = W + '/probe_to_json_p60.py'
CFG_NV = W + '/config_tracker_NvSORT.yml'
CLIP = W + '/Clip-10_h264/clip-01/clip-01.mp4'
GT01 = W + '/Clip-10_h264/clip-01/gt/clip-01_gt_tracking.txt'
BOAT = 8
LOCKS = [W + '/scpilot/p60/p60.lock', W + '/scpilot/p60_coco/p60.lock', W + '/scpilot/p59/p59.lock', W + '/scpilot/p55/p55.lock']

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def stani(msg): sys.exit('!! ' + msg)

print('=== p60c priprema COCO detektora', flush=True)
for lk in LOCKS:
    if os.path.exists(lk):
        pid = open(lk).read().strip()
        if pid and os.path.exists('/proc/' + pid): stani('%s jos radi -- pokreni kad zavrsi.' % lk)
for p in (PT, INFER_SRC, PTJ, CFG_NV, CLIP, GT01):
    if not os.path.exists(p):
        stani('nedostaje %s%s' % (p, '  (yolo11m.pt je skinula p60_detektor.py u /workspace)' if p == PT else
                                   '  (nastaje pri pokretanju p60_fvessel.py)' if p == PTJ else ''))
os.makedirs(C, exist_ok=True)

# ----------------------------------------------------------------------------- 1. ONNX
if os.path.exists(ONNX) and os.path.exists(LAB):
    print('1) ONNX vec postoji: %s' % ONNX, flush=True)
else:
    exp = [p for p in (DSY + '/utils/export_yolo11.py', DSY + '/utils/export_yoloV8.py') if os.path.exists(p)]
    if not exp: stani('u %s/utils nema export_yolo11.py ni export_yoloV8.py' % DSY)
    try:
        import onnx  # noqa
    except ImportError:
        print('1) instaliram paket onnx (potreban za izvoz) ...', flush=True)
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnx'], capture_output=True, text=True)
        if r.returncode != 0: stani('pip install onnx nije uspio:\n' + r.stdout[-800:] + r.stderr[-800:])
    shutil.copy2(PT, C + '/yolo11m.pt')
    for old in glob.glob(C + '/*.onnx') + [C + '/labels.txt']:
        if os.path.exists(old) and old != ONNX: os.remove(old)
    print('1) izvoz u ONNX (%s, ulaz 640) ...' % os.path.basename(exp[0]), flush=True)
    r = subprocess.run([sys.executable, exp[0], '-w', 'yolo11m.pt', '-s', '640'], cwd=C, capture_output=True, text=True)
    nov = [p for p in glob.glob(C + '/*.onnx')]
    if r.returncode != 0 or len(nov) != 1 or not os.path.exists(C + '/labels.txt'):
        stani('izvoz nije uspio (kod %d, onnx: %s):\n%s\n%s' % (r.returncode, nov, r.stdout[-1500:], r.stderr[-2500:]))
    os.replace(nov[0], ONNX); os.replace(C + '/labels.txt', LAB)
    print('   gotovo: %s (%.0f MB)' % (ONNX, os.path.getsize(ONNX) / 1e6), flush=True)
nazivi = [l.strip() for l in open(LAB) if l.strip()]
if len(nazivi) != 80 or nazivi[BOAT] != 'boat': stani('labels_coco.txt nije COCO (80 klasa, 8 = boat): %d klasa, [8] = %s' % (len(nazivi), nazivi[BOAT] if len(nazivi) > BOAT else '-'))

# ----------------------------------------------------------------------------- 2. config
src = open(INFER_SRC).read().splitlines(); out = []; uprop = False; vidjeno = set()
ZAM = {'onnx-file': ONNX, 'model-engine-file': ENG, 'labelfile-path': LAB, 'num-detected-classes': '80'}
IZBACI = {'int8-calib-file', 'filter-out-class-ids', 'custom-network-config', 'model-file', 'tlt-encoded-model', 'tlt-model-key'}
klase_odjeljci = []
for l in src:
    s = l.strip()
    if s.startswith('['):
        if uprop:                                        # kraj [property]: dodaj sto nedostaje
            for k, v in ZAM.items():
                if k not in vidjeno: out.append('%s=%s' % (k, v))
            out.append('filter-out-class-ids=' + ';'.join(str(i) for i in range(80) if i != BOAT))
        uprop = s == '[property]'
        if re.match(r'\[class-attrs-\d+\]', s): klase_odjeljci.append(s)
        out.append(l); continue
    k = s.split('=')[0].strip() if '=' in s and not s.startswith('#') else None
    if uprop and k in ZAM: out.append('%s=%s' % (k, ZAM[k])); vidjeno.add(k); continue
    if uprop and k in IZBACI: continue
    out.append(l)
if uprop:
    for k, v in ZAM.items():
        if k not in vidjeno: out.append('%s=%s' % (k, v))
    out.append('filter-out-class-ids=' + ';'.join(str(i) for i in range(80) if i != BOAT))
open(INFER_NEW, 'w').write('\n'.join(out) + '\n')
print('2) config: %s' % INFER_NEW, flush=True)
for l in out:
    k = l.split('=')[0].strip()
    if k in ('onnx-file', 'model-engine-file', 'labelfile-path', 'num-detected-classes', 'network-mode', 'maintain-aspect-ratio',
             'pre-cluster-threshold', 'nms-iou-threshold', 'cluster-mode', 'custom-lib-path', 'parse-bbox-func-name', 'engine-create-func-name'):
        print('   ' + l.strip(), flush=True)
print('   filter-out-class-ids = sve osim 8 (boat)', flush=True)
if klase_odjeljci: print('   UPOZORENJE: config ima odjeljke %s (vrijede po broju klase) -- javi Claudeu' % klase_odjeljci, flush=True)

# ----------------------------------------------------------------------------- 3. engine
def pokreni(ulaz, izlaz, timeout):
    env = dict(os.environ, MYTRACKER_FPS='25.01'); t = time.time()
    try:
        r = subprocess.run([sys.executable, PTJ, ulaz, CFG_NV, izlaz, INFER_NEW], cwd=W, capture_output=True, text=True, env=env, timeout=timeout)
        return r.returncode, (r.stdout or '') + '\n----\n' + (r.stderr or ''), time.time() - t
    except subprocess.TimeoutExpired:
        return -9, 'prekoraceno vrijeme', time.time() - t
kratko = C + '/kratko_clip01.mp4'
if not os.path.exists(kratko):
    r = subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', CLIP, '-frames:v', '150', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-preset', 'veryfast', '-crf', '18', '-an', kratko], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(kratko): stani('ffmpeg isjecak nije uspio: ' + r.stderr[-800:])
DIRS = [W, DSY, W + '/det_glavni/B_yolo11m/weights', C]
def zasticeno(ulaz, izlaz, timeout, logp):
    """pokreni DeepStream; svi postojeci engine-i se kopiraju prije i vracaju ako ih je nesto prepisalo"""
    stari = sorted({p for d in DIRS for p in glob.glob(d + '/*.engine')})
    sh = {p: sha(p) for p in stari}
    for p in stari: shutil.copy2(p, p + '.p60cbak')
    biljeg = time.time() - 1
    try:
        kod, log, dt = pokreni(ulaz, izlaz, timeout)
        open(logp, 'w').write(log)
    finally:
        for p in stari:
            b = p + '.p60cbak'
            if os.path.exists(p) and sha(p) == sh[p]:
                os.remove(b); continue
            if os.path.exists(p):
                dst = ENG if not os.path.exists(ENG) else p + '.p60c_novi'
                os.replace(p, dst); print('   !! %s je bio prepisan -> novi premjesten u %s' % (p, dst), flush=True)
            os.replace(b, p); print('   %s VRACEN iz sigurnosne kopije' % p, flush=True)
    if not os.path.exists(ENG):
        nv = sorted([p for d in DIRS for p in glob.glob(d + '/*.engine') if p not in sh and os.path.getmtime(p) >= biljeg], key=os.path.getmtime)
        if nv: os.replace(nv[-1], ENG); print('   novi engine %s -> %s' % (nv[-1], ENG), flush=True)
    for p, h in sh.items():
        if not os.path.exists(p) or sha(p) != h: stani('postojeci engine %s nije isti kao prije -- javi Claudeu' % p)
    return kod, dt

if os.path.exists(ENG):
    print('3) engine vec postoji: %s' % ENG, flush=True)
else:
    print('3) gradim TensorRT engine (prvi put 3-10 min; postojeci engine-i su zasticeni kopijom) ...', flush=True)
    kod, dt = zasticeno(kratko, C + '/proba_gradnja.json', 2400, C + '/gradnja_log.txt')
    if not os.path.exists(ENG):
        stani('engine nije nastao (kod %d, %.0f s). Posalji Claudeu: tail -60 %s/gradnja_log.txt' % (kod, dt, C))
    print('   engine gotov (%.0f s, %.0f MB); postojeci engine-i nepromijenjeni' % (dt, os.path.getsize(ENG) / 1e6), flush=True)

# ----------------------------------------------------------------------------- 4. provjera
print('4) provjera: ucitavanje engine-a i detekcije na prvih 150 frejmova clip-01 ...', flush=True)
mt = os.path.getmtime(ENG)
kod, dt = zasticeno(kratko, C + '/proba.json', 900, C + '/proba_log.txt')
if kod != 0 or not os.path.exists(C + '/proba.json'): stani('provjera nije uspjela (kod %d). Posalji Claudeu: tail -60 %s/proba_log.txt' % (kod, C))
if os.path.getmtime(ENG) != mt: print('   UPOZORENJE: engine je ponovno izgraden umjesto ucitan', flush=True)
d = json.load(open(C + '/proba.json'))
if isinstance(d, dict): d = d.get('frames') or d.get('data') or list(d.values())[0]
okv = [(int(fd['frame']), x) for fd in d for x in fd.get('detections', fd.get('objects', []))]
kl = sorted({str(x.get('class', x.get('cls'))) for _, x in okv})
print('   frejmova %d, okvira %d (%.1f po frejmu), klase u izlazu: %s, %.0f FPS s trackerom NvSORT' % (
      len(d), len(okv), len(okv) / max(1, len(d)), kl, len(d) / max(dt, 1e-6)), flush=True)
gt = {}
for l in open(GT01):
    s = l.strip().split(',')
    if len(s) >= 6: gt.setdefault(int(float(s[0])), []).append((int(float(s[1])), [float(q) for q in s[2:6]]))
def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[0] + a[2], b[0] + b[2]); y2 = min(a[1] + a[3], b[1] + b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1); u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u > 0 else 0.0
po = {}
for f, x in okv: po.setdefault(f, []).append(x)
for s in sorted(gt):
    f = int(round(s * 25.01)) + 14
    if f >= 150: break
    L = po.get(f, [])
    opis = ' | '.join('GT%d najbolji IoU %.2f' % (g, max([iou(b, [float(q) for q in (x.get('box') or x.get('bbox'))[:4]]) for x in L] or [0])) for g, b in gt[s])
    print('   sekunda %d (frejm %d): okvira %d | %s' % (s, f, len(L), opis), flush=True)
print('\nGOTOVO. Sad pokreni p60 s COCO detektorom (oko 1,5-2 h):', flush=True)
print('   nohup python3 /workspace/p60_fvessel.py /workspace/Clip-10_h264 --det=coco > /workspace/p60c_out.txt 2>&1 &', flush=True)
