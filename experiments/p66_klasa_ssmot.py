#!/usr/bin/env python3
"""p66 -- NVIDIA trackeri s checkClassMatch: 0 na 16 SSMOT snimaka (glavni postav IEEE rada).

Zasto: p65 je pokazao da na SMD-u (SSMOT detektor) NVIDIA trackeri s checkClassMatch: 0 sustignu vlastiti tracker (NvSORT)
ili ga malo prestignu (NvDCF r1/r2 za ~0,010 HOTA). IEEE rad tvrdi da je vlastiti tracker na SSMOT-u znacajno bolji od NvDeepSORT-a
i oba NvDCF-a -- ali uz referentne NVIDIA postavke (checkClassMatch: 1). Recenzent ce pitati vrijedi li to i bez zabrane klase.

Pokus: isti cjevovod kao p55 (uvozi se /workspace/p55_izvorna_rezolucija.py: streammux 1280x720, YOLO11m, iste kopije probe skripti,
zastita engine-a). JEDINA promjena: kopije 4 NVIDIA yml s checkClassMatch 1 -> 0 u /workspace/p66_cfg/ (izvornici netaknuti).
Vlastiti tracker se NE pusta (koriste se postojeci log_p55_final_ss_<n>.json); p55 na kraju ponovno prevodi polazni track.h
(isti izvorni kod -- kao nakon svakog p55 pokretanja). Logovi: /workspace/scpilot/p66/log_p66_nv_<k>_ss_<n>.json.

Unaprijed zadano (claude/p63_smd_plan.md, odjeljak p66):
 (A) mehanizam na SSMOT-u: HOTA s checkClassMatch 0 znacajno raste (Wilcoxon, Holm preko 4) za >= 3 od 4 NVIDIA trackera
 (B) tvrdnja IEEE rada OSTAJE ako je vlastiti i dalje znacajno bolji po HOTA od NvDeepSORT-a i oba NvDCF-a s checkClassMatch 0;
     inace se doprinos mora preformulirati (prakticna preporuka za DeepStream + probacija)
 Samoprovjera: vlastiti i NVIDIA s checkClassMatch 1 moraju dati brojke iz IEEE rada (HOTA 0,807 / 0,794 / 0,770 / 0,798 / 0,792).

Pokretanje (GPU slobodan, ~25-35 min):  nohup python3 /workspace/p66_klasa_ssmot.py > /workspace/p66_out.txt 2>&1 &
Samo ocjena:                            python3 /workspace/p66_klasa_ssmot.py --analiza
Ako je prekinuto:                       python3 /workspace/p55_izvorna_rezolucija.py --vrati   (pa ponovno pokreni p66 -- nastavlja)
Izlaz:                                  /workspace/scpilot/p66_klasa_ssmot.txt
"""
import os, sys, re, json, shutil, importlib.util
import numpy as np

W = os.environ.get('P55_W', '/workspace'); S = W + '/scpilot'; O = S + '/p66'; CFGD = W + '/p66_cfg'
OUT = S + '/p66_klasa_ssmot.txt'
AN = '--analiza' in sys.argv
if os.path.isdir(W + '/TrackEval') and W + '/TrackEval' not in sys.path: sys.path.insert(0, W + '/TrackEval')
IEEE = {'final': 0.807, 'nv_nvsort': 0.794, 'nv_deep': 0.770, 'nv_dcf1': 0.798, 'nv_dcf2': 0.792}     # Tablica V (p56)

_argv = sys.argv; sys.argv = [_argv[0]]
P55 = W + '/p55_izvorna_rezolucija.py'
if not os.path.exists(P55): sys.exit('!! nema %s' % P55)
_sp = importlib.util.spec_from_file_location('p55m', P55); B = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(B)
sys.argv = _argv
NV0 = list(B.NV); SS16 = list(B.SS16)

def cfg_kopije():
    os.makedirs(CFGD, exist_ok=True); nov = []
    for k, c, lab in NV0:
        t = open(c).read(); rx = re.compile(r'^(\s*checkClassMatch\s*:\s*)1(\b.*)$', re.M)
        if len(rx.findall(t)) != 1: sys.exit('!! u %s nije nadjen tocno jedan "checkClassMatch: 1" -- javi Claudeu' % c)
        t2 = rx.sub(lambda m: m.group(1) + '0' + m.group(2), t)
        a, b = t.splitlines(), t2.splitlines()
        if len(a) != len(b) or sum(1 for x, y in zip(a, b) if x != y) != 1: sys.exit('!! kopija %s se razlikuje u vise od 1 retka' % c)
        dst = CFGD + '/' + os.path.basename(c); open(dst, 'w').write(t2); nov.append((k, dst, lab))
    return nov

def outp66(k, s): return O + '/log_p66_%s_ss_%d.json' % (k, s)
if not AN:
    NV1 = cfg_kopije()
    _outp55 = B.outp
    B.outp = lambda k, s: outp66(k, s) if k.startswith('nv_') else _outp55(k, s)
    B.NV = NV1; B.OWNV = []                                            # vlastiti se ne pusta
    B.O = O; B.LOCK = O + '/p66.lock'
    B.LOCKS = list(B.LOCKS) + [S + '/p55/p55.lock', S + '/p60/p60.lock', S + '/p60_coco/p60.lock', S + '/p64_coco/p60.lock',
                               S + '/p64_ssmot/p60.lock', S + '/p65_klasa0/p60.lock']
    nedostaje = [s for s in SS16 if not B.json_ok(_outp55('final', s))]
    if nedostaje: sys.exit('!! nema log_p55_final_ss_<n>.json za snimke %s -- javi Claudeu' % nedostaje)
    print('p66: NVIDIA trackeri s checkClassMatch: 0 na 16 SSMOT snimaka; konfiguracije u %s; logovi u %s' % (CFGD, O), flush=True)
    B.runs()

# ============================================================================= ocjena (isto kao p55: id_smooth.load_tr/load_gt/pick + TrackEval)
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
    if miss: P('  upozorenje: %s nema sve logove (snimke %s) -- izostavljen' % (lab, miss)); continue
    R[('nv_' + k, 1)] = ev(f1, k + '1'); R[('nv_' + k, 0)] = ev(f0, k + '0'); print('  ocijenjeno: %s' % lab, flush=True)
ks = [(k, lab) for k, _, lab in NV0 if ('nv_' + k, 0) in R]
def mean(r, m): return float(np.mean([r[s][m] for s in SS16]))
P('p66 -- SSMOT 16 snimaka (%s), streammux 1280x720, YOLO11m; NVIDIA trackeri s checkClassMatch 1 (p55) i 0 (p66)' % ','.join(map(str, SS16)))
P(); P('Samoprovjera prema IEEE Tablici V (HOTA): ' + ', '.join('%s %.3f (IEEE %.3f) %s' % (k, mean(R[k] if k == 'final' else R[(k, 1)], 'HOTA'), v,
  'OK' if abs(mean(R[k] if k == 'final' else R[(k, 1)], 'HOTA') - v) < 0.0006 else 'RAZLIKA') for k, v in IEEE.items() if k == 'final' or (k, 1) in R))
P(); P('=== PROSJEK PO SNIMCI (n = 16)')
P('  %-34s %6s %6s %6s %6s %6s %6s' % ('sustav', 'HOTA', 'DetA', 'AssA', 'AssPr', 'IDF1', 'IDSW'))
def red(nm, r): P('  %-34s %6.3f %6.3f %6.3f %6.3f %6.3f %6.1f' % (nm, mean(r, 'HOTA'), mean(r, 'DetA'), mean(r, 'AssA'), mean(r, 'AssPr'), mean(r, 'IDF1'), mean(r, 'IDSW')))
red('vlastiti, konacna', R['final'])
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
            P('  %-5s %-12s razlika %+.4f  veca na %d/16  p = %.4f  Holm %.4f' % (m, lab, d.mean(), int((d > 0).sum()), pv, h))
            sig.setdefault(m, []).append((lab, d.mean(), h))
    return sig
SA = usporedi('=== (A) NVIDIA: checkClassMatch 0 minus 1', lambda k: (R[('nv_' + k, 0)], R[('nv_' + k, 1)]))
SB = usporedi('=== (B) VLASTITI minus NVIDIA s checkClassMatch 0 (Holm preko 4)', lambda k: (R['final'], R[('nv_' + k, 0)]))
nA = sum(1 for lab, d, h in SA['HOTA'] if d > 0 and h < 0.05)
ostaje = all(any(lab == L_ and d > 0 and h < 0.05 for lab, d, h in SB['HOTA']) for L_ in ('NvDeepSORT', 'NvDCF r1', 'NvDCF r2') if any(L_ == x for _, x in ks))
P(); P('Unaprijed zadani kriteriji:')
P('  (A) HOTA s checkClassMatch 0 znacajno raste za %d od %d NVIDIA trackera -> %s' % (nA, len(ks), 'mehanizam VAZI i na SSMOT-u' if nA >= 3 else 'mehanizam na SSMOT-u NIJE potvrden'))
P('  (B) vlastiti %s znacajno bolji po HOTA od NvDeepSORT-a i oba NvDCF-a s checkClassMatch 0 -> %s' % ('JEST' if ostaje else 'NIJE',
  'tvrdnja IEEE rada OSTAJE' if ostaje else 'doprinos treba PREFORMULIRATI'))
P('  razlika vlastiti - NVIDIA (checkClassMatch 0), HOTA: ' + ', '.join('%s %+.4f (Holm %.4f)' % (lab, d, h) for lab, d, h in SB['HOTA']))
open(OUT, 'w').write('\n'.join(L) + '\n'); print('\nrezultat: %s' % OUT)
