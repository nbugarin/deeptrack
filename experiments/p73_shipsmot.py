#!/usr/bin/env python3
"""p73 -- TRACKERS ON ShipsMOT (test split, 40 sequences, identities annotated by hand).

The same pipeline as p60/p64 (imports /workspace/p60_fvessel.py; probe copies probe_moj_p64.py / probe_to_json_p64.py
at the native resolution of the videos). Trackers: ours final + the NVIDIA trackers of p60 (B.NV). Values FROZEN;
nothing is tuned on ShipsMOT. Two detectors (all trackers get the same detections): coco (COCO YOLO11m, class boat, 640)
and ssmot (YOLO11m trained on SSMOT, 640).

Evaluation (set in advance, before any tracker was run):
 - MAIN sample: the 14 fixed-camera sequences found by p71b_shipsmot_camera.py; secondary, descriptive: all sequences
   and the moving-camera ones separately
 - TrackEval on every frame, class-agnostic (all annotated classes are vessel types -> no distractors)
 - ground truth and tracker output are clipped to the image; a box narrower or lower than 1 px after clipping is dropped
 - offset log frame -> GT frame: one for the whole set per detector, searched in -3..+3 on the fixed sequences
 - ours final minus each NVIDIA tracker: two-sided Wilcoxon, Holm over the 4 comparisons per measure (n = 14)
 - detector coverage: if the highest recall (TP / GT) of any tracker on the fixed sequences is < 10 %, the run with that
   detector is reported and discarded
 - rule p58 (v1, v2) on the output of ours, out of sample
Deviation, recorded before the secondary run: the p60 machinery requires one resolution per run, so only the 38 sequences
at 1920x1080 are run (21 at 1280x720 and 115 at 3840x2160 are left out; both are moving-camera sequences).

Run (GPU free; ~1.5-2 h for both detectors, ~35 min for the fixed sequences only):
    nohup python3 /workspace/p73_shipsmot.py > /workspace/p73_out.txt 2>&1 &
Options:              --det=coco | --det=ssmot     --samo-fiksne (run only the 14 fixed sequences)
Evaluation only:      PYTHONPATH=/workspace/TrackEval python3 /workspace/p73_shipsmot.py --analiza
Offset sensitivity:   ... --analiza --det=ssmot --pomak=0   (output p73_shipsmot_ssmot_pomak+0.txt)
If interrupted:       python3 /workspace/p60_fvessel.py --vrati   (then rerun p73 -- it resumes)
Dataset path:         first non-option argument, default /workspace/test_shipmot (the test split, one folder per sequence
                      with NNN.mp4 and gt/gt.txt)
Output:               /workspace/scpilot/p73_shipsmot.txt  (+ scpilot/p73_coco/, scpilot/p73_ssmot/, p73_rezultati.json)
Requires the DeepStream container with Eigen (apt-get install -y libeigen3-dev) to rebuild the library.
"""
import os, sys, json, glob, time, shutil, collections, importlib.util
import numpy as np

W = os.environ.get('P60_W', '/workspace'); S = W + '/scpilot'
AN = '--analiza' in sys.argv
SAMO_FIKSNE = '--samo-fiksne' in sys.argv
DETS = [d for d in ('coco', 'ssmot') if ('--det=' + d) in sys.argv] or ['coco', 'ssmot']
ROOT = next((a for a in sys.argv[1:] if not a.startswith('--')), W + '/test_shipmot')
POMAK = next((int(a.split('=')[1]) for a in sys.argv[1:] if a.startswith('--pomak=')), None)   # forced offset (sensitivity)
_suf = ('_' + '_'.join(DETS) if len(DETS) == 1 else '') + ('' if POMAK is None else '_pomak%+d' % POMAK)
OUT = S + '/p73_shipsmot%s.txt' % _suf; REZJ = S + '/p73_rezultati%s.json' % _suf
FIKSNE = [1, 13, 18, 34, 35, 37, 40, 55, 58, 59, 62, 71, 92, 97]            # p71b, rule set before the measurement
KLASE = {0: 'Container ship', 1: 'General cargo ship', 2: 'Bulk cargo carrier', 3: 'Ferry boat', 4: 'Tug', 5: 'Kayak',
         6: 'Oil carrier', 8: 'Fishing boat', 9: 'Passenger ship', 10: 'Sailing ship', 13: 'Canoe', 14: 'Speed boat'}
DETCFG = {'coco': (W + '/DeepStream-Yolo/config_infer_coco11m_640.txt', W + '/coco_det/yolo11m_coco_640_b1_gpu0_fp16.engine',
                   'general COCO YOLO11m (boat), input 640'),
          'ssmot': (W + '/DeepStream-Yolo/config_infer_11m.txt', None, 'YOLO11m trained on SSMOT, input 640')}
MUX = (1920, 1080)                       # stream multiplexer of the native copies of the probes (videos at 1920x1080)
def ime_od(n): return 'shipsmot_%03d' % n

# ----------------------------------------------------------------------------- p60 as a module (run machinery)
_argv = sys.argv; sys.argv = [_argv[0]]
_sp = importlib.util.spec_from_file_location('p60b', W + '/p60_fvessel.py')
if _sp is None or not os.path.exists(W + '/p60_fvessel.py'): sys.exit('!! missing %s/p60_fvessel.py' % W)
B = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(B)
sys.argv = _argv

def sve_snimke():
    out = []
    for d in sorted(glob.glob(ROOT + '/*'), key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else 10**9):
        b = os.path.basename(d)
        if not b.isdigit(): continue
        v = '%s/%s.mp4' % (d, b); g = d + '/gt/gt.txt'
        if os.path.exists(v) and os.path.exists(g): out.append((ime_od(int(b)), v, g))
    if not out: sys.exit('!! no sequences in %s' % ROOT)
    return out
FIX = {ime_od(n) for n in FIKSNE}
def za_pokretanje():
    # the p60 machinery needs one resolution for all videos -> 21 (720p) and 115 (4K) are not run (both moving-camera, secondary only)
    s = [x for x in sve_snimke() if tuple(B.meta(x[1])[:2]) == MUX]
    return [x for x in s if x[0] in FIX] if SAMO_FIKSNE else s
B.snimke = za_pokretanje
B.PM = W + '/probe_moj_p64.py'; B.PTJ = W + '/probe_to_json_p64.py'
PROT_BASE = [p for p in B.PROT0 if 'coco_det' not in p]
B.LOCKS = list(B.LOCKS) + [S + '/p60/p60.lock', S + '/p60_coco/p60.lock', S + '/p64_coco/p60.lock', S + '/p64_ssmot/p60.lock']

def postavi(d):
    inf, eng, _ = DETCFG[d]
    B.INFER = inf; B.O = S + '/p73_%s' % d; B.LOCK = B.O + '/p60.lock'
    B.PROT0 = PROT_BASE + ([eng] if eng else [])
def spreman(d):
    inf, eng, _ = DETCFG[d]
    return os.path.exists(inf) and (eng is None or os.path.exists(eng))

if not AN:
    sn = za_pokretanje()
    if len(sve_snimke()) != 40: print('!! warning: found %d sequences, expected 40' % len(sve_snimke()), flush=True)
    if not FIX <= {x[0] for x in sve_snimke()}: sys.exit('!! one of the 14 fixed sequences is missing')
    for d in DETS:
        if not spreman(d):
            print('!! detector %s not prepared (%s) -- skipped' % (d, DETCFG[d][0]), flush=True); continue
        print('\n######## p73 ShipsMOT -- detector: %s, %d sequences' % (DETCFG[d][2], len(sn)), flush=True)
        postavi(d); B.runs()

# ============================================================================= EVALUATION
L = []
def P(x=''):
    print(x, flush=True); L.append(str(x))
from scipy.stats import wilcoxon
iou = B.iou

def rez(b, Wk, Hk):
    x0, y0 = max(0.0, b[0]), max(0.0, b[1]); x1, y1 = min(float(Wk), b[0] + b[2]), min(float(Hk), b[1] + b[3])
    return [x0, y0, x1 - x0, y1 - y0] if (x1 - x0) >= 1 and (y1 - y0) >= 1 else None

def ucitaj_gt(g, Wk, Hk):
    G = collections.defaultdict(list); nrez = 0; nizb = 0
    for l in open(g):
        s = l.strip().split(',')
        if len(s) < 6: continue
        t, k = int(float(s[0])), int(float(s[1])); b = [float(q) for q in s[2:6]]
        r = rez(b, Wk, Hk)
        if r is None: nizb += 1; continue
        if r != b: nrez += 1
        G[t].append((k, r))
    return G, nrez, nizb

def te2(items, tag):
    try:
        import trackeval
    except ImportError:
        sys.path.insert(0, W + '/TrackEval'); import trackeval
    TMP = S + '/p73_te_tmp_%s' % tag
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

SN_ALL = sve_snimke()
MT = {}; GT = {}; NREZ = 0; NIZB = 0
for ime, v, g in SN_ALL:
    MT[ime] = B.meta(v); Wk, Hk = MT[ime][0], MT[ime][1]
    GT[ime], a, b = ucitaj_gt(g, Wk, Hk); NREZ += a; NIZB += b
NAZ = dict([('final', 'ours, final'), ('final_v1', 'ours + rule v1'), ('final_v2', 'ours + rule v2')] +
           [('nv_' + k, lab) for k, _, lab in B.NV])
SVE = {}
P('p73 -- ShipsMOT test, %d sequences (%d fixed = main sample), %d frames; IDs annotated by hand; trackers frozen' % (
  len(SN_ALL), len(FIX), sum(MT[i][3] for i, _, _ in SN_ALL)))
P('GT: %d boxes; clipped to the image %d, dropped (< 1 px after clipping) %d' % (
  sum(len(b) for i in GT for b in GT[i].values()), NREZ, NIZB))
neobicne = [(i, MT[i][0], MT[i][1]) for i, _, _ in SN_ALL if (MT[i][0], MT[i][1]) != (1920, 1080)]
if neobicne: P('Resolution other than 1920x1080 -- not run (the p60 machinery needs one resolution; both moving-camera): %s' % neobicne)

for d in DETS:
    postavi(d)
    kljucevi = ['final'] + ['nv_' + k for k, _, _ in B.NV]
    LOG = {}
    for ime, v, g in SN_ALL:
        for k in kljucevi:
            p = B.outp(k, ime)
            if B.json_ok(p): LOG[(k, ime)] = B.ucitaj(p)
    SN = [x for x in SN_ALL if all((k, x[0]) in LOG for k in ('final',))]
    P(); P('#' * 100); P('DETECTOR: %s' % DETCFG[d][2])
    if not FIX <= {x[0] for x in SN}:
        P('  no logs of our tracker for all 14 fixed sequences -- skipped (run without --analiza)'); continue
    imaju = [k for k in kljucevi if all((k, ime) in LOG for ime, _, _ in SN)]
    for k in kljucevi:
        if k not in imaju: P('  warning: %s has no logs for all sequences -- left out' % NAZ[k])
    P('  sequences with logs: %d (fixed %d)' % (len(SN), sum(1 for x in SN if x[0] in FIX)))
    # offset (on the fixed sequences)
    zb = {o: 0.0 for o in range(-3, 4)}
    for ime, v, g in SN:
        if ime not in FIX: continue
        for t in range(1, MT[ime][3] + 1, 5):
            for o in zb:
                for k in ('final',) + tuple(x for x in imaju if x != 'final')[:1]:
                    tl = LOG[(k, ime)].get(t - o, [])
                    if tl: zb[o] += sum(max(iou(b, tb) for (_, tb, _) in tl) for _, b in GT[ime].get(t, []))
    o = max(zb, key=lambda x: (round(zb[x], 3), -abs(x - 1)))
    P('Offset: GT frame = log frame + o; o = %+d (sum of IoU %s)' % (o, ', '.join('%+d: %.0f' % kv for kv in sorted(zb.items()))))
    if POMAK is not None:
        o = POMAK; P('  FORCED OFFSET (sensitivity): o = %+d' % o)
    # rule p58
    for ime, v, g in SN:
        Wk, Hk, fps, n = MT[ime]; T = LOG[('final', ime)]
        LOG[('final_v1', ime)] = B.gated(T, B.pravilo(T, fps, Wk, Hk)); LOG[('final_v2', ime)] = B.gated(T, B.pravilo(T, fps, Wk, Hk, c_big=0.7))
    sustavi = imaju[:1] + ['final_v1', 'final_v2'] + imaju[1:]
    def items_za(k):
        it = {}
        for ime, v, g in SN:
            Wk, Hk, _, n = MT[ime]; T = LOG[(k, ime)]; Ts = {}
            sx, sy = Wk / MUX[0], Hk / MUX[1]          # probe boxes are in stream-multiplexer coordinates (1920x1080)
            for t in range(1, n + 1):
                tl = [(i, rez([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], Wk, Hk)) for (i, b, c) in T.get(t - o, [])]
                tl = [(i, b) for i, b in tl if b is not None]
                if tl: Ts[t] = tl
            it[ime] = ({t: v_ for t, v_ in GT[ime].items() if 1 <= t <= n}, Ts, n)
        return it
    REZ = {}
    for k in sustavi:
        t0 = time.time(); REZ[k] = te2(items_za(k), d)
        print('  evaluated: %s (%.0f s)' % (NAZ[k], time.time() - t0), flush=True)
    SVE[d] = dict(o=o, rez=REZ)
    json.dump(SVE, open(REZJ, 'w'))
    st = B.stanje()
    def grupa(R, k, skup): return [s for s in R[k] if s in skup]
    def mean(R, k, m, skup): return float(np.mean([R[k][s][m] for s in grupa(R, k, skup)]))
    def zbroj(R, k, m, skup): return sum(R[k][s][m] for s in grupa(R, k, skup))
    SVI = {x[0] for x in SN}; POM = SVI - FIX
    # detector coverage (fixed sequences)
    odz = {k: zbroj(REZ, k, 'TP', FIX) / max(1.0, zbroj(REZ, k, 'TP', FIX) + zbroj(REZ, k, 'FN', FIX)) for k in imaju}
    kmax = max(odz, key=odz.get)
    P(); P('Coverage (fixed): highest recall %.1f %% (%s); rule < 10 %% -> %s' % (
      100 * odz[kmax], NAZ[kmax], 'DETECTOR DISCARDED' if odz[kmax] < 0.10 else 'detector kept'))
    def tablica(skup, naslov):
        P(); P('=== %s (n = %d; IDSW, Frag, FP, FN = sum)' % (naslov, len(skup)))
        P('  %-24s %6s %6s %6s %6s %6s %6s %6s %7s %7s %5s' % ('system', 'HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW', 'Frag', 'FP', 'FN', 'FPS'))
        for k in sustavi:
            fv = [st.get('vrijeme', {}).get('%s|%s' % (k, ime)) for ime in skup]; fv = [x for x in fv if x]
            fr = (sum(x[0] for x in fv) / sum(x[1] for x in fv)) if fv else float('nan')
            P('  %-24s %6.3f %6.3f %6.3f %6.3f %6.3f %6.0f %6.0f %7.0f %7.0f %5s' % (NAZ[k], mean(REZ, k, 'HOTA', skup), mean(REZ, k, 'DetA', skup),
              mean(REZ, k, 'AssA', skup), mean(REZ, k, 'IDF1', skup), mean(REZ, k, 'MOTA', skup), zbroj(REZ, k, 'IDSW', skup),
              zbroj(REZ, k, 'Frag', skup), zbroj(REZ, k, 'FP', skup), zbroj(REZ, k, 'FN', skup), '%.0f' % fr if fr == fr else '-'))
    def usporedba(skup, naslov, mjere):
        P(); P(naslov)
        for m in mjere:
            ps = []; rows = []
            for k in imaju[1:]:
                sq = sorted(grupa(REZ, 'final', skup))
                dd = np.array([REZ['final'][s][m] - REZ[k][s][m] for s in sq])
                pv = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
                ps.append(pv); rows.append((k, dd))
            for (k, dd), pv, h in zip(rows, ps, holm(ps)):
                bolje = int((dd > 0).sum()) if m not in ('IDSW', 'Frag') else int((dd < 0).sum())
                P('  %-5s vs %-12s difference %+.3f  better on %d/%d  p = %.4f  Holm %.4f' % (m, NAZ[k], dd.mean(), bolje, len(dd), pv, h))
    tablica(FIX, 'MAIN: FIXED SEQUENCES, mean per sequence')
    usporedba(FIX, '=== MAIN: OURS FINAL minus NVIDIA, fixed (two-sided Wilcoxon, Holm over %d comparisons per measure; n = %d)' % (
      len(imaju) - 1, len(FIX)), ('HOTA', 'IDF1', 'AssA', 'DetA', 'IDSW', 'Frag'))
    P(); P('=== TRACK CONFIRMATION RULE (p58), fixed (out of sample)')
    for k in ('final_v1', 'final_v2'):
        P('  %-24s FP %+6.0f  TP %+6.0f  HOTA %+.4f  IDF1 %+.4f' % (NAZ[k], zbroj(REZ, k, 'FP', FIX) - zbroj(REZ, 'final', 'FP', FIX),
          zbroj(REZ, k, 'TP', FIX) - zbroj(REZ, 'final', 'TP', FIX), mean(REZ, k, 'HOTA', FIX) - mean(REZ, 'final', 'HOTA', FIX),
          mean(REZ, k, 'IDF1', FIX) - mean(REZ, 'final', 'IDF1', FIX)))
    if POM:
        tablica(SVI, 'SECONDARY (descriptive): ALL SEQUENCES')
        tablica(POM, 'SECONDARY (descriptive): MOVING-CAMERA SEQUENCES')
        usporedba(SVI, '=== SECONDARY (descriptive): ours minus NVIDIA, all sequences (n = %d)' % len(SVI), ('HOTA', 'IDF1', 'AssA'))
    P(); P('Per sequence (HOTA, in the order %s):' % ', '.join(NAZ[k] for k in sustavi))
    for ime, _, _ in SN: P('  %-14s %s ' % (ime, 'F' if ime in FIX else 'P') + ' '.join('%.3f' % REZ[k][ime]['HOTA'] for k in sustavi))
    open(OUT, 'w').write('\n'.join(L) + '\n')
P(); P('Note: ShipsMOT test split; main sample the 14 fixed-camera sequences (p71b); vessels mostly visible throughout -> few chances for IDSW.')
open(OUT, 'w').write('\n'.join(L) + '\n')
print('\nresult: %s' % OUT)
