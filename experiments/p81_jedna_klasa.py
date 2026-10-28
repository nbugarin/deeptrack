#!/usr/bin/env python3
"""p81: THE SAME DETECTOR WITHOUT CLASS INFORMATION (H5) -- SMD-Plus VIS_Onshore, 40 sequences, SSMOT detector.

Plan and decision rule written BEFORE the run: docs/preregistration_p81.md (committed before the run).
The same pipeline and trackers as p64 (imports /workspace/p60_fvessel.py: track.h -> final variant and restore, engine
protection, resuming after an interruption). The ONLY change: a pad probe on the sink pad of nvtracker sets class_id = 0
for every detection, so the trackers get the same boxes and confidences as in p64 but no class information.
Trackers (frozen): ours final, NvSORT, NvDeepSORT, NvDCF r1, NvDCF r2. Reference: the p64 logs in scpilot/p64_ssmot.
First: ours final with the UNMODIFIED probe on the first sequence must be identical to p64 (otherwise STOP).
Then: every box in every p81 log must have class 0 (otherwise STOP); indicators recomputed as in p80.
Primary: A = HOTA(ours) - HOTA(NvDCF r1) per sequence; A_multi (p64), A_single (p81); delta = 0.005 HOTA.
  outcome A: advantage not present in p81 (not: mean A_single > delta and Holm < 0.05) AND shrinkage
             D = A_multi - A_single has mean > delta and two-sided Wilcoxon p < 0.05
  outcome B: advantage over NvDCF r1 still present in p81
  outcome C: otherwise

Run (GPU free, about 1 h; the evaluation follows automatically):
    nohup python3 /workspace/p81_jedna_klasa.py >> /workspace/p81_out.txt 2>&1 &
Evaluation only:   python3 /workspace/p81_jedna_klasa.py --analiza
If interrupted:    python3 /workspace/p60_fvessel.py --vrati     (then rerun p81 -- it resumes)
Output:            /workspace/scpilot/p81_jedna_klasa.txt (+ scpilot/p81/p81_sve.json)
"""
import os, re, sys, json, glob, time, shutil, collections, importlib.util, atexit
import numpy as np

W = os.environ.get('P60_W', '/workspace'); S = W + '/scpilot'
AN = '--analiza' in sys.argv
SMD = W + '/SMD_mot'
O = S + '/p81'; O_CHK = O + '/provjera'; O_REF = S + '/p64_ssmot'; O_CM0 = S + '/p65_klasa0'
OUT = S + '/p81_jedna_klasa.txt'; OUTJ = O + '/p81_sve.json'; LOCK81 = O + '/p81.lock'
INFER = W + '/DeepStream-Yolo/config_infer_11m.txt'
PM_SRC1 = W + '/probe_moj_p81_src.py'; PT_SRC1 = W + '/probe_to_json_p81_src.py'
DELTA = 0.005
TAU, T_CLASS, T_CONF = 0.8, 0.01, 0.50              # indicator definitions and thresholds of p80
PRIMARY = 'dcf1'                                     # NvDCF reidType 1
PLOVILA = {'ferry', 'vessel/ship', 'boat', 'kayak', 'sail boat', 'sailboat', 'speed boat'}

# ----------------------------------------------------------------------------- p60 as a module (run machinery)
_argv = sys.argv; sys.argv = [_argv[0]]
if not os.path.exists(W + '/p60_fvessel.py'): sys.exit('!! missing %s/p60_fvessel.py' % W)
_sp = importlib.util.spec_from_file_location('p60b', W + '/p60_fvessel.py'); B = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(B)
sys.argv = _argv
NV0 = list(B.NV)                                     # [(key, yml, name)]
NAZ = dict([('final', 'ours, final')] + [('nv_' + k, lab) for k, _, lab in NV0])

def smd_snimke():
    out = []
    for d in sorted(glob.glob(SMD + '/MVI_*')):
        ime = os.path.basename(d); v = '%s/%s.mp4' % (d, ime); g = d + '/gt/gt.txt'
        if os.path.exists(v) and os.path.exists(g): out.append((ime, v, g))
    if not out: sys.exit('!! no sequences in %s' % SMD)
    return out

# ----------------------------------------------------------------------------- probe copy that removes the class
MARK = re.compile(r'^osd_pad\.add_probe\(.*\)[ \t]*$', re.M)
TRK = re.compile(r"""^tracker\s*=\s*Gst\.ElementFactory\.make\(\s*['"]nvtracker['"]""", re.M)
INJ = '''
# ---- p81: every detection gets class 0 before the tracker (same boxes and confidences; only the class is removed)
def p81_one_class(pad, info, u_data):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not bm:
        return Gst.PadProbeReturn.OK
    lf = bm.frame_meta_list
    while lf:
        try:
            fm = pyds.NvDsFrameMeta.cast(lf.data)
        except StopIteration:
            break
        lo = fm.obj_meta_list
        while lo:
            try:
                ob = pyds.NvDsObjectMeta.cast(lo.data)
            except StopIteration:
                break
            ob.class_id = 0
            try:
                lo = lo.next
            except StopIteration:
                break
        try:
            lf = lf.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK
tracker.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, p81_one_class, 0)
# ---- end p81'''
def napravi_probu(src, dst):
    if not os.path.exists(src): sys.exit('!! missing %s' % src)
    t = open(src).read(); m = list(MARK.finditer(t))
    if len(m) != 1: sys.exit('!! in %s, not exactly one line "osd_pad.add_probe(...)" (found %d) -- report this' % (os.path.basename(src), len(m)))
    if len(TRK.findall(t)) != 1: sys.exit('!! in %s, not exactly one nvtracker element named "tracker" -- report this' % os.path.basename(src))
    t2 = t[:m[0].end()] + INJ + t[m[0].end():]
    if t2.replace(INJ, '', 1) != t: sys.exit('!! the copy of %s differs in more than the inserted block -- report this' % os.path.basename(src))
    open(dst, 'w').write(t2)

def frames(p):
    out = {}
    for fd in json.load(open(p)):
        d = fd.get('detections') or []
        if d: out[int(fd['frame'])] = sorted((int(x['id']), tuple(round(float(v), 2) for v in x['box'][:4])) for x in d)
    return out

def uzmi_lock81():
    os.makedirs(O, exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(LOCK81, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
            atexit.register(pusti_lock81); return
        except FileExistsError:
            try: pid = open(LOCK81).read().strip()
            except FileNotFoundError: continue
            if pid and os.path.exists('/proc/' + pid): sys.exit('!! p81 is already running (PID %s) -- do NOT run twice.' % pid)
            try: os.remove(LOCK81)
            except FileNotFoundError: pass
    sys.exit('!! cannot take the lock %s' % LOCK81)
def pusti_lock81():
    try:
        if open(LOCK81).read().strip() == str(os.getpid()): os.remove(LOCK81)
    except Exception: pass

# ======================================================================= GPU
def gpu():
    uzmi_lock81()
    SN = smd_snimke()
    B.INFER = INFER; B.PROT0 = [p for p in B.PROT0 if 'coco_det' not in p]
    B.LOCKS = list(B.LOCKS) + [S + '/p60/p60.lock', S + '/p60_coco/p60.lock', S + '/p64_coco/p60.lock', S + '/p64_ssmot/p60.lock', S + '/p65_klasa0/p60.lock']
    # 1. reproducibility check against p64
    ime0 = SN[0][0]; ref = '%s/log_p60_final_%s.json' % (O_REF, ime0); chk = '%s/log_p60_final_%s.json' % (O_CHK, ime0)
    if not B.json_ok(ref): sys.exit('!! no p64 log of our tracker for %s (%s)' % (ime0, ref))
    if not B.json_ok(chk):
        print('\n######## p81, step 1: reproducibility check -- ours final, UNMODIFIED probe, %s' % ime0, flush=True)
        B.O = O_CHK; B.LOCK = O_CHK + '/p60.lock'; B.NV = []; B.snimke = lambda: SN[:1]
        B.PM_SRC = W + '/probe_moj.py'; B.PT_SRC = W + '/probe_to_json.py'
        B.PM = W + '/probe_moj_p81_provjera.py'; B.PTJ = W + '/probe_to_json_p81_provjera.py'
        B.runs()
    a, b = frames(chk), frames(ref)
    if a != b:
        fs = set(a) | set(b); ne = sum(1 for f in fs if a.get(f) != b.get(f))
        sys.exit('!! CHECK FAILED: ours final on %s differs from p64 in %d of %d frames -- STOP, report this output.' % (ime0, ne, len(fs)))
    print('reproducibility check: ours final on %s identical to p64 (%d frames with boxes)' % (ime0, len(a)), flush=True)
    # 2. all trackers with the class removed
    napravi_probu(W + '/probe_moj.py', PM_SRC1); napravi_probu(W + '/probe_to_json.py', PT_SRC1)
    print('\n######## p81, step 2: five trackers, class_id = 0 for every detection before the tracker', flush=True)
    B.O = O; B.LOCK = O + '/p60.lock'; B.NV = NV0; B.snimke = smd_snimke
    B.PM_SRC = PM_SRC1; B.PT_SRC = PT_SRC1; B.PM = W + '/probe_moj_p81.py'; B.PTJ = W + '/probe_to_json_p81.py'
    B.runs()

# ======================================================================= EVALUATION
L = []
def P(x=''):
    print(x, flush=True); L.append(str(x))
def holm(ps):
    o = np.argsort(ps); hp = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(ps) - r) * ps[i])); hp[i] = run
    return hp
def wil(d):
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    if not np.any(d != 0): return 1.0
    return float(wilcoxon(d).pvalue)
def first(x, keys):
    for k in keys:
        if k in x and x[k] is not None: return x[k]
    return None
def indicators(paths):
    """as in p80: share of boxes with confidence >= 0.8; class change rate between frames f and f+1 of the same track ID"""
    n_conf = n_hi = n_pair = n_chg = 0; n_box = 0; n_cls_nz = 0
    for p in paths:
        data = json.load(open(p)); last = {}
        for fd in sorted(data, key=lambda d: int(d['frame'])):
            f = int(fd['frame']); cur = {}
            for x in fd.get('detections') or []:
                n_box += 1; c = first(x, ('conf', 'confidence', 'score'))
                if c is not None: n_conf += 1; n_hi += float(c) >= TAU
                k = first(x, ('class', 'class_id', 'classId', 'cls', 'label')); tid = x.get('id')
                if k is not None and int(k) != 0: n_cls_nz += 1
                if k is not None and tid is not None:
                    cur[tid] = k
                    if tid in last and last[tid][0] == f - 1: n_pair += 1; n_chg += last[tid][1] != k
            for tid, k in cur.items(): last[tid] = (f, k)
    return {'boxes': n_box, 'boxes_class_not_0': n_cls_nz, 'conf_share': (n_hi / n_conf) if n_conf else float('nan'),
            'class_change': (n_chg / n_pair) if n_pair else float('nan'), 'pairs': n_pair}
def decision(mean_single, holm_single, mean_shrink, p_shrink):
    present = mean_single > DELTA and holm_single < 0.05
    if present:
        return 'B', ('advantage over NvDCF r1 still present without the class (mean %+.4f > %.3f, Holm %.4f) -> CONTRADICTED: '
                     'the advantage on SMD-Plus does not come from the class indicator; H5 and the detector condition in H reworded' % (mean_single, DELTA, holm_single))
    if mean_shrink > DELTA and p_shrink < 0.05:
        return 'A', ('advantage absent without the class (mean %+.4f, Holm %.4f) and it shrinks by %+.4f (p = %.4f) -> H5 prediction CONFIRMED '
                     'with one detector: removing the class indicator alone removes the advantage' % (mean_single, holm_single, mean_shrink, p_shrink))
    return 'C', ('inconclusive: advantage not present (mean %+.4f, Holm %.4f) but the shrinkage %+.4f (p = %.4f) does not pass the rule '
                 '-> descriptive, H5 status unchanged' % (mean_single, holm_single, mean_shrink, p_shrink))

def analiza():
    from scipy.optimize import linear_sum_assignment
    iou = B.iou
    SN = smd_snimke()
    try: o = int(json.load(open(S + '/p64_rezultati.json'))['ssmot']['o'])
    except Exception: sys.exit('!! cannot read the p64 offset for the SSMOT detector from %s/p64_rezultati.json -- run the p64 evaluation first' % S)
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
        TMP = S + '/p81_te_tmp_%s' % tag
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
                          IDF1=float(r['Identity']['IDF1']), IDSW=float(r['CLEAR']['IDSW']), Frag=float(r['CLEAR']['Frag']),
                          FP=float(r['CLEAR']['CLR_FP']), FN=float(r['CLEAR']['CLR_FN']))
        shutil.rmtree(TMP, ignore_errors=True)
        return res
    GT = {ime: ucitaj_gt(ime) for ime, _, _ in SN}; MT = {ime: B.meta(v) for ime, v, _ in SN}
    def logp(d, k, ime): return '%s/log_p60_%s_%s.json' % (d, k, ime)
    def items(d, k):
        it = {}
        for ime, _, _ in SN:
            p = logp(d, k, ime)
            if not B.json_ok(p): return None
            T = B.ucitaj(p); Gv, Gd = GT[ime]; n = MT[ime][3]; Gs = {}; Ts = {}
            for t in range(1, n + 1):
                gv = Gv.get(t, []); gd = [b for _, b in Gd.get(t, [])]
                if gv: Gs[t] = gv
                tl = ukloni_ometace(T.get(t - o, []), gv, gd)
                if tl: Ts[t] = [(i, b) for (i, b, c) in tl]
            it[ime] = (Gs, Ts, n)
        return it
    KS = ['final'] + ['nv_' + k for k, _, _ in NV0]
    OUTD = {'delta': DELTA, 'offset': o, 'n': len(SN)}
    P('=== p81: the same detector without class information (H5); rule: docs/preregistration_p81.md; delta = %.3f HOTA' % DELTA)
    P('    SMD-Plus VIS_Onshore, %d sequences, SSMOT detector; offset o = %+d (as in p64); reference p64, class removed in p81' % (len(SN), o))
    # --- checks
    ime0 = SN[0][0]; chk = logp(O_CHK, 'final', ime0); ref = logp(O_REF, 'final', ime0)
    if not (B.json_ok(chk) and B.json_ok(ref)): sys.exit('!! reproducibility check log missing (%s) -- run without --analiza' % chk)
    if frames(chk) != frames(ref): sys.exit('!! CHECK FAILED: ours final on %s differs from p64 -- STOP' % ime0)
    P('    check 1, reproducibility: ours final on %s identical to p64' % ime0)
    miss = [(k, ime) for k in KS for ime, _, _ in SN if not B.json_ok(logp(O, k, ime))]
    if miss: sys.exit('!! %d p81 logs missing (first: %s, %s) -- run without --analiza' % (len(miss), NAZ[miss[0][0]], miss[0][1]))
    IND = {}
    for k in KS: IND[k] = indicators([logp(O, k, ime) for ime, _, _ in SN])
    nz = {k: IND[k]['boxes_class_not_0'] for k in KS if IND[k]['boxes_class_not_0']}
    if nz: sys.exit('!! CHECK FAILED: boxes with class != 0 in p81 logs (%s) -- the probe did not act; STOP' % ', '.join('%s %d' % (NAZ[k], v) for k, v in nz.items()))
    P('    check 2, manipulation: all %d boxes in all p81 logs have class 0' % sum(IND[k]['boxes'] for k in KS))
    ref_ind = indicators([logp(O_REF, 'final', ime) for ime, _, _ in SN]); new_ind = IND['final']
    inf_new = (new_ind['class_change'] >= T_CLASS) or (new_ind['conf_share'] >= T_CONF)
    P('    check 3, indicators of ours (p80 definitions): p64 class change %.2f %%, conf >= 0.8 %.1f %% | p81 class change %.2f %%, conf >= 0.8 %.1f %% -> %s' % (
      100 * ref_ind['class_change'], 100 * ref_ind['conf_share'], 100 * new_ind['class_change'], 100 * new_ind['conf_share'],
      'still INFORMATIVE (report this)' if inf_new else 'not informative, as intended'))
    OUTD['indicators'] = {'p64_ours': ref_ind, 'p81_ours': new_ind, 'p81_informative': bool(inf_new)}
    # --- evaluation
    R = {}
    for k in KS:
        R[('multi', k)] = te2(items(O_REF, k), 'm_' + k); R[('single', k)] = te2(items(O, k), 's_' + k)
        if k != 'final':
            it = items(O_CM0, k)
            if it is not None: R[('cm0', k)] = te2(it, 'c_' + k)
        print('  evaluated: %s' % NAZ[k], flush=True)
    seqs = sorted(R[('single', 'final')])
    def mean(r, m): return float(np.mean([r[s][m] for s in seqs]))
    def zbroj(r, m): return sum(r[s][m] for s in seqs)
    P(); P('=== A) MEAN PER SEQUENCE (n = %d; IDSW, Frag, FP, FN = sum over sequences)' % len(seqs))
    P('  %-40s %6s %6s %6s %6s %6s %6s %7s %7s' % ('system', 'HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW', 'Frag', 'FP', 'FN'))
    lab_src = {'multi': 'with class (p64)', 'single': 'class removed (p81)', 'cm0': 'checkClassMatch 0 (p65)'}
    for src in ('multi', 'single', 'cm0'):
        for k in KS:
            if (src, k) not in R: continue
            r = R[(src, k)]
            P('  %-40s %6.3f %6.3f %6.3f %6.3f %6.0f %6.0f %7.0f %7.0f' % ('%s, %s' % (NAZ[k], lab_src[src]), mean(r, 'HOTA'), mean(r, 'DetA'),
              mean(r, 'AssA'), mean(r, 'IDF1'), zbroj(r, 'IDSW'), zbroj(r, 'Frag'), zbroj(r, 'FP'), zbroj(r, 'FN')))
        P()
    OUTD['means'] = {'%s|%s' % kk: {m: mean(R[kk], m) for m in ('HOTA', 'DetA', 'AssA', 'IDF1')} for kk in R}
    OUTD['per_sequence_HOTA'] = {'%s|%s' % kk: {s: R[kk][s]['HOTA'] for s in seqs} for kk in R}
    # --- primary
    nvk = ['nv_' + k for k, _, _ in NV0]
    def razlike(src):
        rows = []
        for k in nvk:
            d = np.array([R[(src, 'final')][s]['HOTA'] - R[(src, k)][s]['HOTA'] for s in seqs]); rows.append((k, d, wil(d)))
        hs = holm([r[2] for r in rows])
        return {k: (d, p, h) for (k, d, p), h in zip(rows, hs)}
    DM, DS = razlike('multi'), razlike('single')
    P('=== B) OURS minus NVIDIA, HOTA (two-sided Wilcoxon, Holm over %d trackers); "advantage present" = mean > %.3f and Holm < 0.05' % (len(nvk), DELTA))
    P('  %-12s | %-40s | %-40s | %s' % ('tracker', 'with class (p64)', 'class removed (p81)', 'shrinkage D = A_multi - A_single'))
    OUTD['contrasts'] = {}
    for k in nvk:
        dm, pm, hm = DM[k]; ds, ps, hs_ = DS[k]; D = dm - ds; pD = wil(D)
        pres = lambda m, h: 'present' if (m > DELTA and h < 0.05) else 'not present'
        P('  %-12s | %+.4f Holm %.4f %2d/%d %-11s | %+.4f Holm %.4f %2d/%d %-11s | %+.4f p = %.4f %2d/%d' % (
          NAZ[k], dm.mean(), hm, int((dm > 0).sum()), len(seqs), pres(dm.mean(), hm), ds.mean(), hs_, int((ds > 0).sum()), len(seqs),
          pres(ds.mean(), hs_), D.mean(), pD, int((D > 0).sum()), len(seqs)))
        OUTD['contrasts'][NAZ[k]] = {'A_multi': float(dm.mean()), 'holm_multi': hm, 'A_single': float(ds.mean()), 'holm_single': hs_,
                                     'p_single': ps, 'D': float(D.mean()), 'p_D': pD}
    dm, _, _ = DM['nv_' + PRIMARY]; ds, _, hsp = DS['nv_' + PRIMARY]; D = dm - ds
    out, txt = decision(float(ds.mean()), hsp, float(D.mean()), wil(D))
    P(); P('=== C) RULE FROM THE PREREGISTRATION (primary: NvDCF r1)'); P('  outcome %s: %s' % (out, txt))
    OUTD['outcome'] = out; OUTD['verdict'] = txt
    # --- secondary
    P(); P('=== D) SECONDARY CONTRASTS (descriptive, without correction)')
    P('  ours with the class removed (= class penalty 0) minus ours in p64 (class penalty 0.5):')
    OUTD['penalty0_minus_penalty05'] = {}
    for m in ('HOTA', 'AssA', 'IDF1', 'DetA', 'IDSW', 'Frag'):
        d = np.array([R[('single', 'final')][s][m] - R[('multi', 'final')][s][m] for s in seqs]); p = wil(d)
        P('    %-5s %+9.4f  p = %.4f  higher on %d/%d' % (m, d.mean(), p, int((d > 0).sum()), len(seqs)))
        OUTD['penalty0_minus_penalty05'][m] = {'mean': float(d.mean()), 'p': p}
    for a, b, naslov in (('single', 'multi', 'NVIDIA with the class removed (p81) minus with class (p64), HOTA:'),
                         ('single', 'cm0', 'NVIDIA with the class removed (p81) minus checkClassMatch 0 (p65), HOTA:')):
        P('  ' + naslov)
        for k in nvk:
            if (b, k) not in R: P('    %-12s no logs' % NAZ[k]); continue
            d = np.array([R[(a, k)][s]['HOTA'] - R[(b, k)][s]['HOTA'] for s in seqs])
            P('    %-12s %+9.4f  p = %.4f  higher on %d/%d' % (NAZ[k], d.mean(), wil(d), int((d > 0).sum()), len(seqs)))
    os.makedirs(O, exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n'); json.dump(OUTD, open(OUTJ, 'w'), indent=1)
    print('\nOutput: %s' % OUT, flush=True)

if __name__ == '__main__':
    if not AN: gpu()
    analiza()
