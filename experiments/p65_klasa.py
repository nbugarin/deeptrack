#!/usr/bin/env python3
"""p65 -- CONTROLLED EXPERIMENT: NVIDIA trackers with checkClassMatch: 0 on SMD-Plus (SSMOT detector).

Question (IEEE v20, VI-C): is the advantage of our tracker on SMD due to the NVIDIA trackers associating only the same class
(checkClassMatch: 1), while the SSMOT detector changes the class 10x more often on unfamiliar vessels (4.37 % vs 0.45 % transitions)?

Experiment: the same 4 NVIDIA trackers, the same settings, the ONLY change checkClassMatch: 1 -> 0 (association regardless of class).
Our tracker is NOT rerun (its logs from p64_ssmot are used; track.h is not touched, nothing is compiled).
Copies of the configurations in /workspace/p65_cfg/ (the original yml files are not changed). Logs: /workspace/scpilot/p65_klasa0/.

Set in advance:
 - main measure: change in HOTA / AssA / IDF1 of each NVIDIA tracker (checkClassMatch 0 minus 1), Wilcoxon, Holm over 4
 - explanation SUPPORTED if for at least 3 of 4 NVIDIA trackers HOTA with checkClassMatch: 0 is significantly higher than with 1
 - additionally: ours minus NVIDIA (checkClassMatch: 0) -- is the class penalty better than ignoring the class completely
 - same evaluation as p64: every frame, distractors removed, offset the same as in the main p64 evaluation for the SSMOT detector

Run (GPU free, ~40 min):  nohup python3 /workspace/p65_klasa.py > /workspace/p65_out.txt 2>&1 &
Evaluation only:          PYTHONPATH=/workspace/TrackEval python3 /workspace/p65_klasa.py --analiza
Output:                   /workspace/scpilot/p65_klasa.txt
"""
import os, sys, re, json, glob, time, shutil, collections, importlib.util
import numpy as np

W = os.environ.get('P60_W', '/workspace'); S = W + '/scpilot'
AN = '--analiza' in sys.argv
SMD = W + '/SMD_mot'; O = S + '/p65_klasa0'; O_REF = S + '/p64_ssmot'; CFGD = W + '/p65_cfg'
OUT = S + '/p65_klasa.txt'
PLOVILA = {'ferry', 'vessel/ship', 'boat', 'kayak', 'sail boat', 'sailboat', 'speed boat'}

_argv = sys.argv; sys.argv = [_argv[0]]
if not os.path.exists(W + '/p60_fvessel.py'): sys.exit('!! missing %s/p60_fvessel.py' % W)
_sp = importlib.util.spec_from_file_location('p60b', W + '/p60_fvessel.py'); B = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(B)
sys.argv = _argv

def smd_snimke():
    out = []
    for d in sorted(glob.glob(SMD + '/MVI_*')):
        ime = os.path.basename(d); v = '%s/%s.mp4' % (d, ime); g = d + '/gt/gt.txt'
        if os.path.exists(v) and os.path.exists(g): out.append((ime, v, g))
    if not out: sys.exit('!! no sequences in %s' % SMD)
    return out
SN = smd_snimke()
NV0 = list(B.NV)                                       # [(key, yml, name)]

# ----------------------------------------------------------------------------- copies of the configurations with checkClassMatch: 0
def pripremi_cfg():
    os.makedirs(CFGD, exist_ok=True); nov = []
    for k, c, lab in NV0:
        t = open(c).read()
        rx = re.compile(r'^(\s*checkClassMatch\s*:\s*)1(\b.*)$', re.M)
        if len(rx.findall(t)) != 1: sys.exit('!! not exactly one "checkClassMatch: 1" found in %s -- report this' % c)
        t2 = rx.sub(lambda m: m.group(1) + '0' + m.group(2), t)
        a, b = t.splitlines(), t2.splitlines(); diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(a) != len(b) or len(diff) != 1: sys.exit('!! copy %s differs in more than 1 line' % c)
        dst = CFGD + '/' + os.path.basename(c); open(dst, 'w').write(t2); nov.append((k, dst, lab))
    return nov
NV1 = pripremi_cfg() if not AN else [(k, CFGD + '/' + os.path.basename(c), lab) for k, c, lab in NV0]

if not AN:
    # our tracker is not rerun: the existing logs from p64_ssmot are copied (runs() then skips them and does not touch track.h)
    os.makedirs(O, exist_ok=True); fali = []
    for ime, _, _ in SN:
        src = O_REF + '/log_p60_final_%s.json' % ime; dst = O + '/log_p60_final_%s.json' % ime
        if not os.path.exists(src): fali.append(ime); continue
        if not os.path.exists(dst): shutil.copy2(src, dst)
    if fali: sys.exit('!! no logs of our tracker from p64_ssmot for: %s' % ', '.join(fali[:5]))
    B.snimke = smd_snimke; B.NV = NV1
    B.PM = W + '/probe_moj_p64.py'; B.PTJ = W + '/probe_to_json_p64.py'
    B.INFER = W + '/DeepStream-Yolo/config_infer_11m.txt'; B.O = O; B.LOCK = O + '/p60.lock'
    B.PROT0 = [p for p in B.PROT0 if 'coco_det' not in p]
    B.LOCKS = list(B.LOCKS) + [S + '/p60/p60.lock', S + '/p60_coco/p60.lock', S + '/p64_coco/p60.lock', S + '/p64_ssmot/p60.lock']
    print('p65: NVIDIA trackers with checkClassMatch: 0 on SMD-Plus (SSMOT detector); configurations in %s' % CFGD, flush=True)
    B.runs()

# ============================================================================= EVALUATION
L = []
def P(x=''):
    print(x, flush=True); L.append(str(x))
from scipy.stats import wilcoxon
from scipy.optimize import linear_sum_assignment
iou = B.iou
KL = {int(k): v for k, v in json.load(open(SMD + '/klase.json')).items()}
def je_plovilo(c): return KL.get(c, '').lower() in PLOVILA
def ucitaj_gt(ime):
    Gv = collections.defaultdict(list); Gd = collections.defaultdict(list)
    for l in open('%s/%s/gt/gt.txt' % (SMD, ime)):
        s = l.strip().split(',')
        if len(s) < 8: continue
        t, k, c = int(s[0]), int(s[1]), int(s[7]); b = [float(q) for q in s[2:6]]
        (Gv if je_plovilo(c) else Gd)[t].append((k, b))
    return Gv, Gd
def ukloni_ometace(tl, gv, gd):
    if not tl or not gd: return tl
    G = [b for _, b in gv] + list(gd); M = np.array([[iou(t[1], g) for g in G] for t in tl])
    r, c = linear_sum_assignment(-M); van = {i for i, j in zip(r, c) if M[i, j] >= 0.5 and j >= len(gv)}
    return [t for i, t in enumerate(tl) if i not in van]
def te2(items, tag):
    try:
        import trackeval
    except ImportError:
        sys.path.insert(0, W + '/TrackEval'); import trackeval
    TMP = S + '/p65_te_tmp_%s' % tag
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
                      IDF1=float(r['Identity']['IDF1']), IDSW=float(r['CLEAR']['IDSW']), Frag=float(r['CLEAR']['Frag']))
    shutil.rmtree(TMP, ignore_errors=True)
    return res
def holm(ps):
    o = np.argsort(ps); hp = [0] * len(ps); run = 0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); hp[i] = run
    return hp

o = -1                                                   # main p64 offset for the SSMOT detector
try:
    o = int(json.load(open(S + '/p64_rezultati.json'))['ssmot']['o'])
except Exception:
    pass
GT = {ime: ucitaj_gt(ime) for ime, _, _ in SN}; MT = {ime: B.meta(v) for ime, v, _ in SN}
def items(logdir, k):
    it = {}
    for ime, _, _ in SN:
        p = '%s/log_p60_%s_%s.json' % (logdir, k, ime)
        if not B.json_ok(p): return None
        T = B.ucitaj(p); Gv, Gd = GT[ime]; n = MT[ime][3]; Gs = {}; Ts = {}
        for t in range(1, n + 1):
            gv = Gv.get(t, []); gd = [b for _, b in Gd.get(t, [])]
            if gv: Gs[t] = gv
            tl = ukloni_ometace(T.get(t - o, []), gv, gd)
            if tl: Ts[t] = [(i, b) for (i, b, c) in tl]
        it[ime] = (Gs, Ts, n)
    return it
P('p65 -- SMD-Plus VIS_Onshore, %d sequences, SSMOT detector; NVIDIA trackers with checkClassMatch: 1 (p64) and 0 (p65); offset o = %+d' % (len(SN), o))
R = {}
it = items(O_REF, 'final')
if it is None: sys.exit('!! no logs of our tracker in %s' % O_REF)
R['ours'] = te2(it, 'ours')
for k, _, lab in NV0:
    a = items(O_REF, 'nv_' + k); b = items(O, 'nv_' + k)
    if a is None or b is None: P('  warning: %s does not have all logs (%s)' % (lab, 'checkClassMatch 1' if a is None else 'checkClassMatch 0')); continue
    R[(k, 1)] = te2(a, k + '1'); R[(k, 0)] = te2(b, k + '0'); print('  evaluated: %s' % lab, flush=True)
ks = [(k, lab) for k, _, lab in NV0 if (k, 0) in R]
seqs = sorted(R['ours'])
def mean(r, m): return float(np.mean([r[s][m] for s in seqs]))
def zbroj(r, m): return sum(r[s][m] for s in seqs)
P(); P('=== MEAN PER SEQUENCE (n = %d; IDSW, Frag = sum)' % len(seqs))
P('  %-34s %6s %6s %6s %6s %6s %6s' % ('system', 'HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW', 'Frag'))
P('  %-34s %6.3f %6.3f %6.3f %6.3f %6.0f %6.0f' % ('ours, final', mean(R['ours'], 'HOTA'), mean(R['ours'], 'DetA'), mean(R['ours'], 'AssA'),
  mean(R['ours'], 'IDF1'), zbroj(R['ours'], 'IDSW'), zbroj(R['ours'], 'Frag')))
for k, lab in ks:
    for c in (1, 0):
        r = R[(k, c)]
        P('  %-34s %6.3f %6.3f %6.3f %6.3f %6.0f %6.0f' % ('%s, checkClassMatch %d' % (lab, c), mean(r, 'HOTA'), mean(r, 'DetA'), mean(r, 'AssA'),
          mean(r, 'IDF1'), zbroj(r, 'IDSW'), zbroj(r, 'Frag')))
def usporedi(naslov, par):
    P(); P(naslov)
    sig = {}
    for m in ('HOTA', 'AssA', 'IDF1', 'DetA', 'IDSW'):
        ps = []; rows = []
        for k, lab in ks:
            a, b = par(k)
            d = np.array([a[s][m] - b[s][m] for s in seqs]); pv = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            ps.append(pv); rows.append((lab, d))
        for (lab, d), pv, h in zip(rows, ps, holm(ps)):
            P('  %-5s %-12s difference %+.3f  higher on %d/%d  p = %.4f  Holm %.4f' % (m, lab, d.mean(), int((d > 0).sum()), len(d), pv, h))
            sig.setdefault(m, []).append((lab, d.mean(), h))
    return sig
S1 = usporedi('=== NVIDIA: checkClassMatch 0 minus 1 (is the ban on associating different classes the cause?)', lambda k: (R[(k, 0)], R[(k, 1)]))
S2 = usporedi('=== OURS minus NVIDIA with checkClassMatch 0 (class penalty versus ignoring the class completely)', lambda k: (R['ours'], R[(k, 0)]))
n_bolji = sum(1 for lab, d, h in S1['HOTA'] if d > 0 and h < 0.05)
P(); P('Criterion set in advance: explanation SUPPORTED if HOTA with checkClassMatch 0 rises significantly for >= 3 of 4 NVIDIA trackers.')
P('  -> rises significantly for %d of %d: %s' % (n_bolji, len(ks), 'SUPPORTED' if n_bolji >= 3 else 'NOT SUPPORTED'))
P('  remaining difference ours - NVIDIA (checkClassMatch 0), HOTA: ' + ', '.join('%s %+.3f (Holm %.4f)' % (lab, d, h) for lab, d, h in S2['HOTA']))
open(OUT, 'w').write('\n'.join(L) + '\n'); print('\nresult: %s' % OUT)
