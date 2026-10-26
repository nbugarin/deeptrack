#!/usr/bin/env python3
"""p66 -- NVIDIA trackers with checkClassMatch: 0 on 16 SSMOT sequences (main setup of the IEEE paper).

Why: p65 showed that on SMD (SSMOT detector) the NVIDIA trackers with checkClassMatch: 0 catch up with our tracker (NvSORT)
or slightly overtake it (NvDCF r1/r2 by ~0.010 HOTA). The IEEE paper claims that our tracker on SSMOT is significantly better than NvDeepSORT
and both NvDCF -- but with the reference NVIDIA settings (checkClassMatch: 1). A reviewer will ask whether this also holds without the class ban.

Experiment: the same pipeline as p55 (imports /workspace/p55_izvorna_rezolucija.py: streammux 1280x720, YOLO11m, the same copies of the probe scripts,
engine protection). The ONLY change: copies of the 4 NVIDIA yml files with checkClassMatch 1 -> 0 in /workspace/p66_cfg/ (originals untouched).
Our tracker is NOT rerun (the existing log_p55_final_ss_<n>.json are used); at the end p55 recompiles the baseline track.h
(the same source code -- as after every p55 run). Logs: /workspace/scpilot/p66/log_p66_nv_<k>_ss_<n>.json.

Set in advance:
 (A) mechanism on SSMOT: HOTA with checkClassMatch 0 rises significantly (Wilcoxon, Holm over 4) for >= 3 of 4 NVIDIA trackers
 (B) the claim of the IEEE paper STANDS if ours is still significantly better in HOTA than NvDeepSORT and both NvDCF with checkClassMatch 0;
     otherwise the contribution must be reworded (practical recommendation for DeepStream + probation)
 Self-check: ours and NVIDIA with checkClassMatch 1 must give the numbers from the IEEE paper (HOTA 0.807 / 0.794 / 0.770 / 0.798 / 0.792).

Run (GPU free, ~25-35 min):  nohup python3 /workspace/p66_klasa_ssmot.py > /workspace/p66_out.txt 2>&1 &
Evaluation only:             python3 /workspace/p66_klasa_ssmot.py --analiza
If interrupted:              python3 /workspace/p55_izvorna_rezolucija.py --vrati   (then rerun p66 -- it resumes)
Output:                      /workspace/scpilot/p66_klasa_ssmot.txt
"""
import os, sys, re, json, shutil, importlib.util
import numpy as np

W = os.environ.get('P55_W', '/workspace'); S = W + '/scpilot'; O = S + '/p66'; CFGD = W + '/p66_cfg'
OUT = S + '/p66_klasa_ssmot.txt'
AN = '--analiza' in sys.argv
if os.path.isdir(W + '/TrackEval') and W + '/TrackEval' not in sys.path: sys.path.insert(0, W + '/TrackEval')
IEEE = {'final': 0.807, 'nv_nvsort': 0.794, 'nv_deep': 0.770, 'nv_dcf1': 0.798, 'nv_dcf2': 0.792}     # Table V (p56)

_argv = sys.argv; sys.argv = [_argv[0]]
P55 = W + '/p55_izvorna_rezolucija.py'
if not os.path.exists(P55): sys.exit('!! missing %s' % P55)
_sp = importlib.util.spec_from_file_location('p55m', P55); B = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(B)
sys.argv = _argv
NV0 = list(B.NV); SS16 = list(B.SS16)

def cfg_kopije():
    os.makedirs(CFGD, exist_ok=True); nov = []
    for k, c, lab in NV0:
        t = open(c).read(); rx = re.compile(r'^(\s*checkClassMatch\s*:\s*)1(\b.*)$', re.M)
        if len(rx.findall(t)) != 1: sys.exit('!! not exactly one "checkClassMatch: 1" found in %s -- report this' % c)
        t2 = rx.sub(lambda m: m.group(1) + '0' + m.group(2), t)
        a, b = t.splitlines(), t2.splitlines()
        if len(a) != len(b) or sum(1 for x, y in zip(a, b) if x != y) != 1: sys.exit('!! copy %s differs in more than 1 line' % c)
        dst = CFGD + '/' + os.path.basename(c); open(dst, 'w').write(t2); nov.append((k, dst, lab))
    return nov

def outp66(k, s): return O + '/log_p66_%s_ss_%d.json' % (k, s)
if not AN:
    NV1 = cfg_kopije()
    _outp55 = B.outp
    B.outp = lambda k, s: outp66(k, s) if k.startswith('nv_') else _outp55(k, s)
    B.NV = NV1; B.OWNV = []                                            # our tracker is not rerun
    B.O = O; B.LOCK = O + '/p66.lock'
    B.LOCKS = list(B.LOCKS) + [S + '/p55/p55.lock', S + '/p60/p60.lock', S + '/p60_coco/p60.lock', S + '/p64_coco/p60.lock',
                               S + '/p64_ssmot/p60.lock', S + '/p65_klasa0/p60.lock']
    nedostaje = [s for s in SS16 if not B.json_ok(_outp55('final', s))]
    if nedostaje: sys.exit('!! no log_p55_final_ss_<n>.json for sequences %s -- report this' % nedostaje)
    print('p66: NVIDIA trackers with checkClassMatch: 0 on 16 SSMOT sequences; configurations in %s; logs in %s' % (CFGD, O), flush=True)
    B.runs()

# ============================================================================= evaluation (same as p55: id_smooth.load_tr/load_gt/pick + TrackEval)
import trackeval
from scipy.stats import wilcoxon
L = []
def P(x=''):
    print(x, flush=True); L.append(str(x))
H = {}; exec(open(W + '/id_smooth.py').read().split('CFG={')[0].split('"""')[2], H)
load_tr, load_gt, pick = H['load_tr'], H['load_gt'], H['pick']
ec = trackeval.Evaluator.get_default_eval_config()
ec.update(PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False, TIME_PROGRESS=False,
          USE_PARALLEL=False, DISPLAY_LESS_PROGRESS=True)
rows = lambda d: sorted((int(f), int(o['id']), *map(float, o['box'][:4])) for f, Lst in d.items() for o in Lst)
def ev(files, tag):
    base = O + '/te_tmp_' + tag; shutil.rmtree(base, ignore_errors=True); si = {}
    for s, p in files.items():
        tr = load_tr(p); g = load_gt(B.GT % s, pick(B.GT % s, tr))
        G = rows(g); R_ = rows(tr); seq = 'S%02d' % s; si[seq] = max([r[0] for r in G + R_] + [1])
        os.makedirs('%s/gt/%s/gt' % (base, seq), exist_ok=True); os.makedirs(base + '/tr/t/data', exist_ok=True)
        with open('%s/gt/%s/gt/gt.txt' % (base, seq), 'w') as f:
            for x in G: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,1,1\n' % x)
        with open('%s/tr/t/data/%s.txt' % (base, seq), 'w') as f:
            for x in R_: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n' % x)
    dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dc.update(GT_FOLDER=base + '/gt', TRACKERS_FOLDER=base + '/tr', BENCHMARK='b', SPLIT_TO_EVAL='', SEQ_INFO=si, TRACKERS_TO_EVAL=['t'], DO_PREPROC=False,
              SKIP_SPLIT_FOL=True, GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt', TRACKER_SUB_FOLDER='data', PRINT_CONFIG=False)
    M = [trackeval.metrics.HOTA({'PRINT_CONFIG': False}), trackeval.metrics.CLEAR({'PRINT_CONFIG': False}), trackeval.metrics.Identity({'PRINT_CONFIG': False})]
    out, _ = trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)], M); Rt = out['MotChallenge2DBox']['t']; shutil.rmtree(base, ignore_errors=True)
    res = {}
    for q in si:
        hh = Rt[q]['pedestrian']; m = lambda k: float(np.mean(hh['HOTA'][k]))
        res[int(q[1:])] = {'HOTA': m('HOTA'), 'DetA': m('DetA'), 'AssA': m('AssA'), 'AssPr': m('AssPr'), 'IDF1': float(hh['Identity']['IDF1']),
                           'IDSW': float(hh['CLEAR']['IDSW'])}
    return res
def holm(ps):
    o = np.argsort(ps); hp = [0] * len(ps); run = 0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); hp[i] = run
    return hp
os.makedirs(O, exist_ok=True)
R = {'final': ev({s: B.outp('final', s) for s in SS16}, 'final')}
for k, _, lab in NV0:
    f1 = {s: W + '/log_p55_nv_%s_ss_%d.json' % (k, s) for s in SS16}; f0 = {s: outp66('nv_' + k, s) for s in SS16}
    miss = [s for s in SS16 if not B.json_ok(f0[s]) or not B.json_ok(f1[s])]
    if miss: P('  warning: %s does not have all logs (sequences %s) -- left out' % (lab, miss)); continue
    R[('nv_' + k, 1)] = ev(f1, k + '1'); R[('nv_' + k, 0)] = ev(f0, k + '0'); print('  evaluated: %s' % lab, flush=True)
ks = [(k, lab) for k, _, lab in NV0 if ('nv_' + k, 0) in R]
def mean(r, m): return float(np.mean([r[s][m] for s in SS16]))
P('p66 -- SSMOT 16 sequences (%s), streammux 1280x720, YOLO11m; NVIDIA trackers with checkClassMatch 1 (p55) and 0 (p66)' % ','.join(map(str, SS16)))
P(); P('Self-check against IEEE Table V (HOTA): ' + ', '.join('%s %.3f (IEEE %.3f) %s' % (k, mean(R[k] if k == 'final' else R[(k, 1)], 'HOTA'), v,
  'OK' if abs(mean(R[k] if k == 'final' else R[(k, 1)], 'HOTA') - v) < 0.0006 else 'DIFFERENT') for k, v in IEEE.items() if k == 'final' or (k, 1) in R))
P(); P('=== MEAN PER SEQUENCE (n = 16)')
P('  %-34s %6s %6s %6s %6s %6s %6s' % ('system', 'HOTA', 'DetA', 'AssA', 'AssPr', 'IDF1', 'IDSW'))
def red(nm, r): P('  %-34s %6.3f %6.3f %6.3f %6.3f %6.3f %6.1f' % (nm, mean(r, 'HOTA'), mean(r, 'DetA'), mean(r, 'AssA'), mean(r, 'AssPr'), mean(r, 'IDF1'), mean(r, 'IDSW')))
red('ours, final', R['final'])
for k, lab in ks:
    red('%s, checkClassMatch 1' % lab, R[('nv_' + k, 1)]); red('%s, checkClassMatch 0' % lab, R[('nv_' + k, 0)])
def usporedi(naslov, par):
    P(); P(naslov); sig = {}
    for m in ('HOTA', 'AssA', 'IDF1', 'DetA', 'IDSW'):
        ps = []; rr = []
        for k, lab in ks:
            a, b = par(k); d = np.array([a[s][m] - b[s][m] for s in SS16]); pv = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            ps.append(pv); rr.append((lab, d))
        for (lab, d), pv, h in zip(rr, ps, holm(ps)):
            P('  %-5s %-12s difference %+.4f  higher on %d/16  p = %.4f  Holm %.4f' % (m, lab, d.mean(), int((d > 0).sum()), pv, h))
            sig.setdefault(m, []).append((lab, d.mean(), h))
    return sig
SA = usporedi('=== (A) NVIDIA: checkClassMatch 0 minus 1', lambda k: (R[('nv_' + k, 0)], R[('nv_' + k, 1)]))
SB = usporedi('=== (B) OURS minus NVIDIA with checkClassMatch 0 (Holm over 4)', lambda k: (R['final'], R[('nv_' + k, 0)]))
nA = sum(1 for lab, d, h in SA['HOTA'] if d > 0 and h < 0.05)
ostaje = all(any(lab == L_ and d > 0 and h < 0.05 for lab, d, h in SB['HOTA']) for L_ in ('NvDeepSORT', 'NvDCF r1', 'NvDCF r2') if any(L_ == x for _, x in ks))
P(); P('Criteria set in advance:')
P('  (A) HOTA with checkClassMatch 0 rises significantly for %d of %d NVIDIA trackers -> %s' % (nA, len(ks), 'mechanism HOLDS on SSMOT too' if nA >= 3 else 'mechanism NOT confirmed on SSMOT'))
P('  (B) ours %s significantly better in HOTA than NvDeepSORT and both NvDCF with checkClassMatch 0 -> %s' % ('IS' if ostaje else 'IS NOT',
  'claim of the IEEE paper STANDS' if ostaje else 'contribution must be REWORDED'))
P('  difference ours - NVIDIA (checkClassMatch 0), HOTA: ' + ', '.join('%s %+.4f (Holm %.4f)' % (lab, d, h) for lab, d, h in SB['HOTA']))
open(OUT, 'w').write('\n'.join(L) + '\n'); print('\nresult: %s' % OUT)
