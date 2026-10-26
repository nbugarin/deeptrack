#!/usr/bin/env python3
"""p64 -- TRACKERI NA SMD-Plus VIS_Onshore (40 snimki, anotacija SVAKOG frejma, ID-ovi izvedeni u p63).

Isti cjevovod kao p60 (uvozi se /workspace/p60_fvessel.py: track.h -> konacna varijanta i vracanje, zastita engine-a,
kopije probe skripti, nastavljanje nakon prekida). Trackeri: vlastiti konacna + NvSORT, NvDeepSORT, NvDCF r1, NvDCF r2.
Vrijednosti trackera ZAMRZNUTE; nista se ne podesava na SMD-u. Dva detektora (svi trackeri dobivaju iste detekcije):
  coco  = opci COCO YOLO11m, samo klasa boat, ulaz 640 (pripremljen s p60_coco_priprema.py)  -- GLAVNI
  ssmot = YOLO11m treniran na SSMOT-u, ulaz 640                                                -- dodatno
(p63: odziv na SMD-u COCO 640 64,5 % / preciznost 87,9 %; SSMOT 640 32,2 % / 77,2 %)

Ocjena (unaprijed zadano, claude/p63_smd_plan.md):
 - TrackEval na SVAKOM frejmu: HOTA, DetA, AssA, IDF1, MOTA, IDSW, Frag, FP, FN; prosjek po snimci (n = 40)
 - ometaci (Buoy, Other): okvir trackera koji se madjarskim uparivanjem s cijelim GT-om (IoU >= 0,5) upari s ometacem se
   uklanja (kao MOT17); GT ometaca se ne broji
 - pomak log-frejm -> GT-frejm: jedan za cijeli skup, trazi se u -3..+3 (ocekivano +1: log broji od 0, GT od 1)
 - vlastiti konacna minus svaki NVIDIA tracker: dvostrani Wilcoxon, Holm preko 4 usporedbe po mjeri
 - osjetljivost: bez MVI_1584 i MVI_1615 (> 5 % dvosmislenih okvira pri izvodjenju ID-ova)
 - samo pokretni brodovi (MotionType = Moving vecina okvira traga; usidreni postaju ometaci)
 - pravilo p58 (v1, v2) na izlazu vlastitog: dFP, dTP, dHOTA -- test izvan uzorka

Pokretanje (GPU slobodan; ~1,5-2 h za oba detektora):
    nohup python3 /workspace/p64_smd.py > /workspace/p64_out.txt 2>&1 &
Samo jedan detektor:  --det=coco  ili  --det=ssmot
Samo ocjena:          PYTHONPATH=/workspace/TrackEval python3 /workspace/p64_smd.py --analiza
Osjetljivost pomaka:  ... --analiza --pomak=1   (isti pomak za oba detektora; izlaz p64_smd_pomak+1.txt, glavni rezultat ostaje)
Ako je prekinuto:     python3 /workspace/p60_fvessel.py --vrati     (pa ponovno pokreni p64 -- nastavlja)
Izlaz:                /workspace/scpilot/p64_smd.txt  (+ scpilot/p64_coco/, scpilot/p64_ssmot/)
"""
import os, sys, json, glob, time, shutil, collections, importlib.util
import numpy as np

W = os.environ.get('P60_W', '/workspace'); S = W + '/scpilot'
AN = '--analiza' in sys.argv
DETS = [d for d in ('coco', 'ssmot') if ('--det=' + d) in sys.argv] or ['coco', 'ssmot']
SMD = W + '/SMD_mot'
ROOT_MAT = next((a for a in sys.argv[1:] if not a.startswith('--')), W + '/SMD_Plus/VIS_Onshore')
POMAK = next((int(a.split('=')[1]) for a in sys.argv[1:] if a.startswith('--pomak=')), None)   # prisilni pomak (osjetljivost)
OUT = S + ('/p64_smd.txt' if POMAK is None else '/p64_smd_pomak%+d.txt' % POMAK)
REZJ = S + ('/p64_rezultati.json' if POMAK is None else '/p64_rezultati_pomak%+d.json' % POMAK)
PLOVILA = {'ferry', 'vessel/ship', 'boat', 'kayak', 'sail boat', 'sailboat', 'speed boat'}
DVOSMISLENE = ['MVI_1584_VIS', 'MVI_1615_VIS']
DETCFG = {'coco': (W + '/DeepStream-Yolo/config_infer_coco11m_640.txt', W + '/coco_det/yolo11m_coco_640_b1_gpu0_fp16.engine',
                   'opci COCO YOLO11m (boat), ulaz 640'),
          'ssmot': (W + '/DeepStream-Yolo/config_infer_11m.txt', None, 'YOLO11m treniran na SSMOT-u, ulaz 640')}

# ----------------------------------------------------------------------------- p60 kao modul (strojevi za pokretanje)
_argv = sys.argv; sys.argv = [_argv[0]]
_sp = importlib.util.spec_from_file_location('p60b', W + '/p60_fvessel.py')
if _sp is None or not os.path.exists(W + '/p60_fvessel.py'): sys.exit('!! nema %s/p60_fvessel.py' % W)
B = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(B)
sys.argv = _argv

def smd_snimke():
    out = []
    for d in sorted(glob.glob(SMD + '/MVI_*')):
        ime = os.path.basename(d); v = '%s/%s.mp4' % (d, ime); g = d + '/gt/gt.txt'
        if os.path.exists(v) and os.path.exists(g): out.append((ime, v, g))
    if not out: sys.exit('!! nema snimki u %s -- prvo pokreni p63_smd_gt.py' % SMD)
    return out
B.snimke = smd_snimke
B.PM = W + '/probe_moj_p64.py'; B.PTJ = W + '/probe_to_json_p64.py'
PROT_BASE = [p for p in B.PROT0 if 'coco_det' not in p]
B.LOCKS = list(B.LOCKS) + [S + '/p60/p60.lock', S + '/p60_coco/p60.lock']

def postavi(d):
    inf, eng, _ = DETCFG[d]
    B.INFER = inf; B.O = S + '/p64_%s' % d; B.LOCK = B.O + '/p60.lock'
    B.PROT0 = PROT_BASE + ([eng] if eng else [])
def spreman(d):
    inf, eng, _ = DETCFG[d]
    return os.path.exists(inf) and (eng is None or os.path.exists(eng))

if not AN:
    for d in DETS:
        if not spreman(d):
            print('!! detektor %s nije pripremljen (%s) -- preskacem%s' % (d, DETCFG[d][0], '; prvo pokreni p60_coco_priprema.py' if d == 'coco' else ''), flush=True)
            continue
        print('\n######## p64 SMD -- detektor: %s' % DETCFG[d][2], flush=True)
        postavi(d); B.runs()

# ============================================================================= OCJENA
L = []
def P(x=''):
    print(x, flush=True); L.append(str(x))
from scipy.stats import wilcoxon
from scipy.optimize import linear_sum_assignment
iou = B.iou
KL = {int(k): v for k, v in json.load(open(SMD + '/klase.json')).items()}
def je_plovilo(c): return KL.get(c, '').lower() in PLOVILA
SN = smd_snimke()

def ucitaj_gt(ime):
    Gv = collections.defaultdict(list); Gd = collections.defaultdict(list); red = collections.defaultdict(list)
    for l in open('%s/%s/gt/gt.txt' % (SMD, ime)):
        s = l.strip().split(',')
        if len(s) < 8: continue
        t, k, c = int(s[0]), int(s[1]), int(s[7]); b = [float(q) for q in s[2:6]]
        red[t].append(k)
        (Gv if je_plovilo(c) else Gd)[t].append((k, b))
    return Gv, Gd, red
def pokretni(ime, red):
    """trag je pokretan ako je vecina njegovih okvira MotionType = Moving (iz .mat, isti redoslijed redaka)"""
    m = glob.glob(ROOT_MAT + '/**/%s_ObjectGT.mat' % ime, recursive=True)
    if not m: return None
    import scipy.io as sio
    S_ = sio.loadmat(m[0], squeeze_me=True, struct_as_record=False)['structXML'].ravel(); gl = collections.defaultdict(list)
    for t, e in enumerate(S_, 1):
        mt = np.atleast_1d(np.asarray(getattr(e, 'MotionType', []), dtype=object)).ravel().tolist() if hasattr(e, 'MotionType') else []
        for k, x in zip(red.get(t, []), mt): gl[k].append(str(x) == 'Moving')
    return {k: (np.mean(v) > 0.5) for k, v in gl.items()}

def ukloni_ometace(tl, gv, gd):
    """madjarsko uparivanje okvira trackera s cijelim GT-om (plovila + ometaci); uklanjaju se upareni s ometacem"""
    if not tl or not gd: return tl
    G = [b for _, b in gv] + list(gd); M = np.array([[iou(t[1], g) for g in G] for t in tl])
    r, c = linear_sum_assignment(-M); van = {i for i, j in zip(r, c) if M[i, j] >= 0.5 and j >= len(gv)}
    return [t for i, t in enumerate(tl) if i not in van]

def te2(items, tag):
    try:
        import trackeval
    except ImportError:
        sys.path.insert(0, W + '/TrackEval'); import trackeval
    TMP = S + '/p64_te_tmp_%s' % tag
    if os.path.exists(TMP): shutil.rmtree(TMP)
    si = {}
    for seq, (G, T, n) in items.items():
        si[seq] = n
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
                      IDF1=float(r['Identity']['IDF1']), MOTA=float(r['CLEAR']['MOTA']), IDSW=float(r['CLEAR']['IDSW']),
                      Frag=float(r['CLEAR']['Frag']), FP=float(r['CLEAR']['CLR_FP']), FN=float(r['CLEAR']['CLR_FN']), TP=float(r['CLEAR']['CLR_TP']))
    shutil.rmtree(TMP, ignore_errors=True)
    return res

def holm(ps):
    o = np.argsort(ps); hp = [0] * len(ps); run = 0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); hp[i] = run
    return hp

GT = {}; MT = {}; MOV = {}
for ime, v, g in SN:
    GT[ime] = ucitaj_gt(ime); MT[ime] = B.meta(v); MOV[ime] = pokretni(ime, GT[ime][2])
NAZ = dict([('final', 'vlastiti, konacna'), ('final_v1', 'vlastiti + pravilo v1'), ('final_v2', 'vlastiti + pravilo v2')] +
           [('nv_' + k, lab) for k, _, lab in B.NV])
SVE = {}
P('p64 -- SMD-Plus VIS_Onshore, %d snimki, %d frejmova; ID-ovi izvedeni (p63); trackeri zamrznuti (IEEE konacna varijanta)' % (
  len(SN), sum(MT[i][3] for i, _, _ in SN)))
nv_ok = sum(len(b) for i in GT for b in GT[i][0].values()); nd_ok = sum(len(b) for i in GT for b in GT[i][1].values())
P('GT: %d okvira plovila, %d okvira ometaca (Buoy, Other); pokretnih tragova %s' % (nv_ok, nd_ok,
  sum(sum(1 for x in (MOV[i] or {}).values() if x) for i in MOV) if all(MOV[i] is not None for i in MOV) else 'nepoznato (nema .mat)'))
for d in DETS:
    postavi(d)
    kljucevi = ['final'] + ['nv_' + k for k, _, _ in B.NV]
    LOG = {}
    for ime, v, g in SN:
        for k in kljucevi:
            p = B.outp(k, ime)
            if B.json_ok(p): LOG[(k, ime)] = B.ucitaj(p)
    imaju = [k for k in kljucevi if all((k, ime) in LOG for ime, _, _ in SN)]
    P(); P('#' * 100); P('DETEKTOR: %s' % DETCFG[d][2])
    if 'final' not in imaju:
        P('  nema logova vlastitog trackera za sve snimke -- preskacem (pokreni bez --analiza)'); continue
    for k in kljucevi:
        if k not in imaju: P('  upozorenje: %s nema logove za sve snimke -- izostavljen' % NAZ[k])
    # pomak
    zb = {o: 0.0 for o in range(-3, 4)}
    for ime, v, g in SN:
        Gv = GT[ime][0]
        for t in range(1, MT[ime][3] + 1, 5):
            for o in zb:
                for k in ('final',) + tuple(x for x in imaju if x != 'final')[:1]:
                    tl = LOG[(k, ime)].get(t - o, [])
                    if tl: zb[o] += sum(max(iou(b, tb) for (_, tb, _) in tl) for _, b in Gv.get(t, []))
    o = max(zb, key=lambda x: (round(zb[x], 3), -abs(x - 1)))
    P('Pomak: GT frejm = log frejm + o; o = %+d (zbroj IoU %s)' % (o, ', '.join('%+d: %.0f' % kv for kv in sorted(zb.items()))))
    if POMAK is not None:
        o = POMAK; P('  PRISILNI POMAK (osjetljivost): o = %+d' % o)
    # pravilo p58
    for ime, v, g in SN:
        Wk, Hk, fps, n = MT[ime]; T = LOG[('final', ime)]
        LOG[('final_v1', ime)] = B.gated(T, B.pravilo(T, fps, Wk, Hk)); LOG[('final_v2', ime)] = B.gated(T, B.pravilo(T, fps, Wk, Hk, c_big=0.7))
    sustavi = imaju[:1] + ['final_v1', 'final_v2'] + imaju[1:]
    def items_za(k, samo_pokretni=False, bez=()):
        it = {}
        for ime, v, g in SN:
            if ime in bez: continue
            Gv, Gd, _ = GT[ime]; n = MT[ime][3]; T = LOG[(k, ime)]; mv = MOV[ime] or {}
            Gs = {}; Ts = {}
            for t in range(1, n + 1):
                gv = Gv.get(t, []); gd = [b for _, b in Gd.get(t, [])]
                if samo_pokretni:
                    gd = gd + [b for kk, b in gv if not mv.get(kk, False)]; gv = [(kk, b) for kk, b in gv if mv.get(kk, False)]
                if gv: Gs[t] = gv
                tl = ukloni_ometace(T.get(t - o, []), gv, gd)
                if tl: Ts[t] = [(i, b) for (i, b, c) in tl]
            it[ime] = (Gs, Ts, n)
        return it
    REZ = {}; REZM = {}
    for k in sustavi:
        t0 = time.time(); REZ[k] = te2(items_za(k), d)
        if k in ('final',) + tuple(imaju[1:]) and all(MOV[i] is not None for i in MOV):
            REZM[k] = te2(items_za(k, samo_pokretni=True), d)
        print('  ocijenjeno: %s (%.0f s)' % (NAZ[k], time.time() - t0), flush=True)
    SVE[d] = dict(o=o, rez=REZ, rez_pokretni=REZM)
    json.dump(SVE, open(REZJ, 'w'))
    st = B.stanje()
    def mean(R, k, m, bez=()): return float(np.mean([R[k][s][m] for s in R[k] if s not in bez]))
    def zbroj(R, k, m): return sum(R[k][s][m] for s in R[k])
    P(); P('=== PROSJEK PO SNIMCI (n = %d; IDSW, Frag, FP, FN = zbroj)' % len(SN))
    P('  %-24s %6s %6s %6s %6s %6s %6s %6s %7s %7s %5s' % ('sustav', 'HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW', 'Frag', 'FP', 'FN', 'FPS'))
    for k in sustavi:
        fv = [st.get('vrijeme', {}).get('%s|%s' % (k, ime)) for ime, _, _ in SN]; fv = [x for x in fv if x]
        fr = (sum(x[0] for x in fv) / sum(x[1] for x in fv)) if fv else float('nan')
        P('  %-24s %6.3f %6.3f %6.3f %6.3f %6.3f %6.0f %6.0f %7.0f %7.0f %5s' % (NAZ[k], mean(REZ, k, 'HOTA'), mean(REZ, k, 'DetA'), mean(REZ, k, 'AssA'),
          mean(REZ, k, 'IDF1'), mean(REZ, k, 'MOTA'), zbroj(REZ, k, 'IDSW'), zbroj(REZ, k, 'Frag'), zbroj(REZ, k, 'FP'), zbroj(REZ, k, 'FN'),
          '%.0f' % fr if fr == fr else '-'))
    def usporedba(R, naslov, mjere, bez=()):
        P(); P(naslov)
        for m in mjere:
            ps = []; rows = []
            for k in imaju[1:]:
                sq = sorted(s for s in R['final'] if s not in bez)
                dd = np.array([R['final'][s][m] - R[k][s][m] for s in sq])
                pv = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
                ps.append(pv); rows.append((k, dd))
            for (k, dd), pv, h in zip(rows, ps, holm(ps)):
                bolje = int((dd > 0).sum()) if m not in ('IDSW', 'Frag') else int((dd < 0).sum())
                P('  %-5s vs %-12s razlika %+.3f  bolje na %d/%d  p = %.4f  Holm %.4f' % (m, NAZ[k], dd.mean(), bolje, len(dd), pv, h))
    usporedba(REZ, '=== VLASTITI KONACNA minus NVIDIA (dvostrani Wilcoxon po snimkama, Holm preko %d usporedbe po mjeri; n = %d)' % (len(imaju) - 1, len(SN)),
              ('HOTA', 'IDF1', 'AssA', 'DetA', 'IDSW', 'Frag'))
    usporedba(REZ, '=== OSJETLJIVOST: bez %s (n = %d)' % (' i '.join(DVOSMISLENE), sum(1 for i, _, _ in SN if i not in DVOSMISLENE)), ('HOTA', 'IDF1', 'AssA'), bez=DVOSMISLENE)
    if REZM:
        P(); P('=== SAMO POKRETNI BRODOVI (usidreni = ometaci)')
        for k in [x for x in sustavi if x in REZM]:
            P('  %-24s HOTA %.3f  AssA %.3f  IDF1 %.3f  IDSW %d' % (NAZ[k], mean(REZM, k, 'HOTA'), mean(REZM, k, 'AssA'), mean(REZM, k, 'IDF1'), zbroj(REZM, k, 'IDSW')))
        usporedba(REZM, '  usporedba (samo pokretni):', ('HOTA', 'IDF1', 'AssA'))
    P(); P('=== PRAVILO POTVRDE TRAGA (p58) NA SMD-u (izvan uzorka)')
    for k in ('final_v1', 'final_v2'):
        P('  %-24s FP %+6.0f  TP %+6.0f  HOTA %+.4f  IDF1 %+.4f' % (NAZ[k], zbroj(REZ, k, 'FP') - zbroj(REZ, 'final', 'FP'),
          zbroj(REZ, k, 'TP') - zbroj(REZ, 'final', 'TP'), mean(REZ, k, 'HOTA') - mean(REZ, 'final', 'HOTA'), mean(REZ, k, 'IDF1') - mean(REZ, 'final', 'IDF1')))
    P(); P('Po snimci (HOTA, redom %s):' % ', '.join(NAZ[k] for k in sustavi))
    for ime, _, _ in SN: P('  %-18s ' % ime + ' '.join('%.3f' % REZ[k][ime]['HOTA'] for k in sustavi))
    open(OUT, 'w').write('\n'.join(L) + '\n')
P(); P('Napomena: ID-ovi u GT-u izvedeni automatski (p63), ne rucno; snimke kratke (prosjek ~16 s).')
open(OUT, 'w').write('\n'.join(L) + '\n')
print('\nrezultat: %s' % OUT)
