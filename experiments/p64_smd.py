#!/usr/bin/env python3
"""p64 -- TRACKERS ON SMD-Plus VIS_Onshore (40 sequences, annotation of EVERY frame, IDs derived in p63).

The same pipeline as p60 (imports /workspace/p60_fvessel.py: track.h -> final variant and restore, engine protection,
copies of the probe scripts, resuming after an interruption). Trackers: ours final + NvSORT, NvDeepSORT, NvDCF r1, NvDCF r2.
Tracker values FROZEN; nothing is tuned on SMD. Two detectors (all trackers get the same detections):
  coco  = general COCO YOLO11m, class boat only, input 640 (prepared with p60_coco_priprema.py)  -- MAIN
  ssmot = YOLO11m trained on SSMOT, input 640                                                    -- additional
(p63: recall on SMD COCO 640 64.5 % / precision 87.9 %; SSMOT 640 32.2 % / 77.2 %)

Evaluation (set in advance, claude/p63_smd_plan.md):
 - TrackEval on EVERY frame: HOTA, DetA, AssA, IDF1, MOTA, IDSW, Frag, FP, FN; mean per sequence (n = 40)
 - distractors (Buoy, Other): a tracker box that Hungarian matching with the whole GT (IoU >= 0.5) matches to a distractor is
   removed (as in MOT17); distractor GT is not counted
 - offset log frame -> GT frame: one for the whole set, searched in -3..+3 (expected +1: the log counts from 0, the GT from 1)
 - ours final minus each NVIDIA tracker: two-sided Wilcoxon, Holm over 4 comparisons per measure
 - sensitivity: without MVI_1584 and MVI_1615 (> 5 % ambiguous boxes when deriving the IDs)
 - moving vessels only (MotionType = Moving for most boxes of the track; anchored ones become distractors)
 - rule p58 (v1, v2) on the output of ours: dFP, dTP, dHOTA -- out-of-sample test

Run (GPU free; ~1.5-2 h for both detectors):
    nohup python3 /workspace/p64_smd.py > /workspace/p64_out.txt 2>&1 &
One detector only:      --det=coco  or  --det=ssmot
Evaluation only:        PYTHONPATH=/workspace/TrackEval python3 /workspace/p64_smd.py --analiza
Offset sensitivity:     ... --analiza --pomak=1   (same offset for both detectors; output p64_smd_pomak+1.txt, the main result stays)
If interrupted:         python3 /workspace/p60_fvessel.py --vrati     (then rerun p64 -- it resumes)
Output:                 /workspace/scpilot/p64_smd.txt  (+ scpilot/p64_coco/, scpilot/p64_ssmot/)
"""
import os, sys, json, glob, time, shutil, collections, importlib.util
import numpy as np

W = os.environ.get('P60_W', '/workspace'); S = W + '/scpilot'
AN = '--analiza' in sys.argv
DETS = [d for d in ('coco', 'ssmot') if ('--det=' + d) in sys.argv] or ['coco', 'ssmot']
SMD = W + '/SMD_mot'
ROOT_MAT = next((a for a in sys.argv[1:] if not a.startswith('--')), W + '/SMD_Plus/VIS_Onshore')
POMAK = next((int(a.split('=')[1]) for a in sys.argv[1:] if a.startswith('--pomak=')), None)   # forced offset (sensitivity)
OUT = S + ('/p64_smd.txt' if POMAK is None else '/p64_smd_pomak%+d.txt' % POMAK)
REZJ = S + ('/p64_rezultati.json' if POMAK is None else '/p64_rezultati_pomak%+d.json' % POMAK)
PLOVILA = {'ferry', 'vessel/ship', 'boat', 'kayak', 'sail boat', 'sailboat', 'speed boat'}
DVOSMISLENE = ['MVI_1584_VIS', 'MVI_1615_VIS']
DETCFG = {'coco': (W + '/DeepStream-Yolo/config_infer_coco11m_640.txt', W + '/coco_det/yolo11m_coco_640_b1_gpu0_fp16.engine',
                   'general COCO YOLO11m (boat), input 640'),
          'ssmot': (W + '/DeepStream-Yolo/config_infer_11m.txt', None, 'YOLO11m trained on SSMOT, input 640')}

# ----------------------------------------------------------------------------- p60 as a module (run machinery)
_argv = sys.argv; sys.argv = [_argv[0]]
_sp = importlib.util.spec_from_file_location('p60b', W + '/p60_fvessel.py')
if _sp is None or not os.path.exists(W + '/p60_fvessel.py'): sys.exit('!! missing %s/p60_fvessel.py' % W)
B = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(B)
sys.argv = _argv

def smd_snimke():
    out = []
    for d in sorted(glob.glob(SMD + '/MVI_*')):
        ime = os.path.basename(d); v = '%s/%s.mp4' % (d, ime); g = d + '/gt/gt.txt'
        if os.path.exists(v) and os.path.exists(g): out.append((ime, v, g))
    if not out: sys.exit('!! no sequences in %s -- run p63_smd_gt.py first' % SMD)
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
            print('!! detector %s not prepared (%s) -- skipping%s' % (d, DETCFG[d][0], '; run p60_coco_priprema.py first' if d == 'coco' else ''), flush=True)
            continue
        print('\n######## p64 SMD -- detector: %s' % DETCFG[d][2], flush=True)
        postavi(d); B.runs()

# ============================================================================= EVALUATION
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
    """a track is moving if most of its boxes have MotionType = Moving (from .mat, same row order)"""
    m = glob.glob(ROOT_MAT + '/**/%s_ObjectGT.mat' % ime, recursive=True)
    if not m: return None
    import scipy.io as sio
    S_ = sio.loadmat(m[0], squeeze_me=True, struct_as_record=False)['structXML'].ravel(); gl = collections.defaultdict(list)
    for t, e in enumerate(S_, 1):
        mt = np.atleast_1d(np.asarray(getattr(e, 'MotionType', []), dtype=object)).ravel().tolist() if hasattr(e, 'MotionType') else []
        for k, x in zip(red.get(t, []), mt): gl[k].append(str(x) == 'Moving')
    return {k: (np.mean(v) > 0.5) for k, v in gl.items()}

def ukloni_ometace(tl, gv, gd):
    """Hungarian matching of tracker boxes with the whole GT (vessels + distractors); those matched to a distractor are removed"""
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
NAZ = dict([('final', 'ours, final'), ('final_v1', 'ours + rule v1'), ('final_v2', 'ours + rule v2')] +
           [('nv_' + k, lab) for k, _, lab in B.NV])
SVE = {}
P('p64 -- SMD-Plus VIS_Onshore, %d sequences, %d frames; IDs derived (p63); trackers frozen (IEEE final variant)' % (
  len(SN), sum(MT[i][3] for i, _, _ in SN)))
nv_ok = sum(len(b) for i in GT for b in GT[i][0].values()); nd_ok = sum(len(b) for i in GT for b in GT[i][1].values())
P('GT: %d vessel boxes, %d distractor boxes (Buoy, Other); moving tracks %s' % (nv_ok, nd_ok,
  sum(sum(1 for x in (MOV[i] or {}).values() if x) for i in MOV) if all(MOV[i] is not None for i in MOV) else 'unknown (no .mat)'))
for d in DETS:
    postavi(d)
    kljucevi = ['final'] + ['nv_' + k for k, _, _ in B.NV]
    LOG = {}
    for ime, v, g in SN:
        for k in kljucevi:
            p = B.outp(k, ime)
            if B.json_ok(p): LOG[(k, ime)] = B.ucitaj(p)
    imaju = [k for k in kljucevi if all((k, ime) in LOG for ime, _, _ in SN)]
    P(); P('#' * 100); P('DETECTOR: %s' % DETCFG[d][2])
    if 'final' not in imaju:
        P('  no logs of our tracker for all sequences -- skipping (run without --analiza)'); continue
    for k in kljucevi:
        if k not in imaju: P('  warning: %s does not have logs for all sequences -- left out' % NAZ[k])
    # offset
    zb = {o: 0.0 for o in range(-3, 4)}
    for ime, v, g in SN:
        Gv = GT[ime][0]
        for t in range(1, MT[ime][3] + 1, 5):
            for o in zb:
                for k in ('final',) + tuple(x for x in imaju if x != 'final')[:1]:
                    tl = LOG[(k, ime)].get(t - o, [])
                    if tl: zb[o] += sum(max(iou(b, tb) for (_, tb, _) in tl) for _, b in Gv.get(t, []))
    o = max(zb, key=lambda x: (round(zb[x], 3), -abs(x - 1)))
    P('Offset: GT frame = log frame + o; o = %+d (sum of IoU %s)' % (o, ', '.join('%+d: %.0f' % kv for kv in sorted(zb.items()))))
    if POMAK is not None:
        o = POMAK; P('  FORCED OFFSET (sensitivity): o = %+d' % o)
    # rule p58
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
        print('  evaluated: %s (%.0f s)' % (NAZ[k], time.time() - t0), flush=True)
    SVE[d] = dict(o=o, rez=REZ, rez_pokretni=REZM)
    json.dump(SVE, open(REZJ, 'w'))
    st = B.stanje()
    def mean(R, k, m, bez=()): return float(np.mean([R[k][s][m] for s in R[k] if s not in bez]))
    def zbroj(R, k, m): return sum(R[k][s][m] for s in R[k])
    P(); P('=== MEAN PER SEQUENCE (n = %d; IDSW, Frag, FP, FN = sum)' % len(SN))
    P('  %-24s %6s %6s %6s %6s %6s %6s %6s %7s %7s %5s' % ('system', 'HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW', 'Frag', 'FP', 'FN', 'FPS'))
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
                P('  %-5s vs %-12s difference %+.3f  better on %d/%d  p = %.4f  Holm %.4f' % (m, NAZ[k], dd.mean(), bolje, len(dd), pv, h))
    usporedba(REZ, '=== OURS FINAL minus NVIDIA (two-sided Wilcoxon over sequences, Holm over %d comparisons per measure; n = %d)' % (len(imaju) - 1, len(SN)),
              ('HOTA', 'IDF1', 'AssA', 'DetA', 'IDSW', 'Frag'))
    usporedba(REZ, '=== SENSITIVITY: without %s (n = %d)' % (' i '.join(DVOSMISLENE), sum(1 for i, _, _ in SN if i not in DVOSMISLENE)), ('HOTA', 'IDF1', 'AssA'), bez=DVOSMISLENE)
    if REZM:
        P(); P('=== MOVING VESSELS ONLY (anchored = distractors)')
        for k in [x for x in sustavi if x in REZM]:
            P('  %-24s HOTA %.3f  AssA %.3f  IDF1 %.3f  IDSW %d' % (NAZ[k], mean(REZM, k, 'HOTA'), mean(REZM, k, 'AssA'), mean(REZM, k, 'IDF1'), zbroj(REZM, k, 'IDSW')))
        usporedba(REZM, '  comparison (moving only):', ('HOTA', 'IDF1', 'AssA'))
    P(); P('=== TRACK CONFIRMATION RULE (p58) ON SMD (out of sample)')
    for k in ('final_v1', 'final_v2'):
        P('  %-24s FP %+6.0f  TP %+6.0f  HOTA %+.4f  IDF1 %+.4f' % (NAZ[k], zbroj(REZ, k, 'FP') - zbroj(REZ, 'final', 'FP'),
          zbroj(REZ, k, 'TP') - zbroj(REZ, 'final', 'TP'), mean(REZ, k, 'HOTA') - mean(REZ, 'final', 'HOTA'), mean(REZ, k, 'IDF1') - mean(REZ, 'final', 'IDF1')))
    P(); P('Per sequence (HOTA, in order %s):' % ', '.join(NAZ[k] for k in sustavi))
    for ime, _, _ in SN: P('  %-18s ' % ime + ' '.join('%.3f' % REZ[k][ime]['HOTA'] for k in sustavi))
    open(OUT, 'w').write('\n'.join(L) + '\n')
P(); P('Note: the IDs in the GT were derived automatically (p63), not by hand; the sequences are short (mean ~16 s).')
open(OUT, 'w').write('\n'.join(L) + '\n')
print('\nresult: %s' % OUT)
