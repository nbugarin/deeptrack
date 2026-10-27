#!/usr/bin/env python3
"""p75 -- RE-ID ON / OFF WITHIN THE SAME CONFIGURATION (16 SSMOT sequences, YOLO11m, same pipeline as p44 = MDPI Tables 4-5)

Reason (27.10.2026): the exchange of the DataAssociator section in MDPI Section 5.6 is not a clean ablation. The NvDCF file has
no minMatchingScore4ReidSimilarity and no matchingScoreWeight4ReidSimilarity, so the library default (0.0) applies:
"+ NvDCF DataAssociator" (-88 % of the excess switches) also switches off the Re-ID term of NvDeepSORT. For the same reason
NvDCF with reidType 1 may not use Re-ID in association at all, although it computes it in every frame.

PLAN WRITTEN DOWN BEFORE THE RUN (27.10.2026, about 16:20). Nothing is selected on these results.
Variants (everything else identical to the original file; only the listed keys change):
  deep_w0   NvDeepSORT, DataAssociator: minMatchingScore4ReidSimilarity 0, matchingScoreWeight4ReidSimilarity 0  (Re-ID term off)
  dcf1_r0   NvDCF r1, ReID: reidType 0                                         (Re-ID module off)
  dcf1_w    NvDCF r1, DataAssociator: min...ReidSimilarity 0.7505, ...Weight4ReidSimilarity 0.3197 (the NvDeepSORT values;
            Re-ID actually in the NvDCF association cost)
  dcf2_r0   NvDCF r2, ReID: reidType 0  (secondary: visual tracker on, output not bit-reproducible)
  deep_rep, dcf1_rep  original files again on SSMOT 1 and 3 -> must be bit-identical to the p44 logs (otherwise STOP)
Comparisons with the p44 logs of the same 16 sequences (log_ss11m_<cfg>_<N>.json); TrackEval as in p44.
Questions and rules:
  P1 (main) Does switching off the Re-ID term remove the excess switches of NvDeepSORT?
      share = (IDSW deep - IDSW deep_w0) / (IDSW deep - IDSW dcf1), means over sequences; Wilcoxon deep_w0 vs deep, vs dcf1, vs dcf2 (Holm over 3)
      share < 25 % AND deep_w0 significantly more IDSW than dcf1 and dcf2  -> "the excess does not come from the Re-ID term" (claim stands, with this evidence)
      share >= 50 %                                                       -> "the Re-ID term is a necessary ingredient of the excess" (rewrite abstract, C6, 6.2, 8)
      otherwise                                                           -> "partial" (descriptive)
  P2  Does NvDCF r1 use Re-ID in association?  dcf1_r0 vs dcf1: number of sequences with bit-identical output.
      16/16 identical -> "NvDCF r1 does not use Re-ID" (rewrite "applies Re-ID in every frame" and "same appearance engine")
  P3  Does Re-ID in the NvDCF association cost increase the switches?  dcf1_w vs dcf1, IDSW, Wilcoxon (one main comparison)
  P2b only if dcf1_r0 fails or differs from dcf1: dcf1_ra0 (enableReAssoc 0) and dcf1_r0b (enableReAssoc 0, reidType 0);
      16/16 identical -> in NvDCF r1, Re-ID is used only in re-association, not in association
  P4  (descriptive) dcf2_r0 vs dcf2: role of Re-ID in re-association
Run (container with DeepStream, GPU free):  nohup python3 /workspace/p75_reid_ablacija.py > /workspace/p75_out.txt 2>&1 &
Monitoring:                                tail -3 /workspace/p75_out.txt
Evaluation only:                           python3 /workspace/p75_reid_ablacija.py --eval
Output: /workspace/scpilot/p75_reid_ablacija.txt (+ scpilot/p75/p75_rezultati.json). If interrupted -> run again, it resumes.
"""
import os, re, sys, json, time, hashlib, subprocess
import numpy as np
for a, b in (('float', float), ('int', int), ('bool', bool)):
    if not hasattr(np, a): setattr(np, a, b)
W = os.environ.get('P75_W', '/workspace'); S = W + '/scpilot'; O = S + '/p75'; LOCK = O + '/p75.lock'
OUT = S + '/p75_reid_ablacija.txt'
PROBE = W + '/probe_to_json.py'; INFER = W + '/DeepStream-Yolo/config_infer_11m.txt'
ENG = W + '/det_glavni/B_yolo11m/weights/best_11m_b1_gpu0_fp16.engine'
BASE = {'nvsort': W + '/config_tracker_NvSORT.yml', 'deep': W + '/config_tracker_maritime.yml',
        'dcf1': W + '/cfg_dcf_novis.yml', 'dcf2': W + '/cfg_nvdcf.yml'}
NAME = {'nvsort': 'NvSORT', 'deep': 'NvDeepSORT', 'dcf1': 'NvDCF r1', 'dcf2': 'NvDCF r2',
        'deep_w0': 'NvDeepSORT, Re-ID w=0', 'dcf1_r0': 'NvDCF r1, reidType 0', 'dcf1_w': 'NvDCF r1 + Re-ID in cost',
        'dcf2_r0': 'NvDCF r2, reidType 0', 'dcf1_ra0': 'NvDCF r1, no re-association', 'dcf1_r0b': 'NvDCF r1, no re-assoc., reidType 0', 'deep_rep': 'NvDeepSORT (again)', 'dcf1_rep': 'NvDCF r1 (again)'}
DA = 'DataAssociator'
VAR = [('deep_rep', 'deep', {}), ('dcf1_rep', 'dcf1', {}),
       ('deep_w0', 'deep', {DA: {'minMatchingScore4ReidSimilarity': '0.0', 'matchingScoreWeight4ReidSimilarity': '0.0'}}),
       ('dcf1_r0', 'dcf1', {'ReID': {'reidType': '0'}}),
       ('dcf1_w', 'dcf1', {DA: {'minMatchingScore4ReidSimilarity': '0.7505', 'matchingScoreWeight4ReidSimilarity': '0.3197'}}),
       ('dcf2_r0', 'dcf2', {'ReID': {'reidType': '0'}})]
TM = 'TrajectoryManagement'
# only if dcf1_r0 fails or differs from dcf1: separates association from re-association (both without re-association)
VAR2 = [('dcf1_ra0', 'dcf1', {TM: {'enableReAssoc': '0'}}),
        ('dcf1_r0b', 'dcf1', {TM: {'enableReAssoc': '0'}, 'ReID': {'reidType': '0'}})]
SS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]; REP_SS = [1, 3]
VID = W + '/SSMOT_video/video/SSMOT_%d.mp4'; GT = W + '/SSMOT_video/ground-truth/SSMOT_%d/gt/gt.txt'
OLD = W + '/log_ss11m_%s_%d.json'            # p44 (MDPI Tables 4-5)
NEW = O + '/log_%s_%d.json'
T_RUN = 1200
REINSTALL = ('apt-get update && apt-get install -y libeigen3-dev && pip install --no-deps git+https://github.com/JonathonLuiten/TrackEval.git '
             '&& python3 -c "import scipy, cv2, trackeval; print(\'ALL OK\')"')
L = []
def P(x=''):
    print(x, flush=True); L.append(str(x))
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def json_ok(p):
    try:
        d = json.load(open(p)); return isinstance(d, list) and len(d) > 0
    except Exception:
        return False
def seqs(v): return REP_SS if v.endswith('_rep') else SS
def tail(r): return (r.stdout or '')[-2500:] + '\n----\n' + (r.stderr or '')[-3500:]

# ---------------- configurations: only the listed keys change, everything else byte for byte the same
def set_key(txt, sec, key, val):
    lines = txt.split('\n')
    hs = [i for i, l in enumerate(lines) if re.match(r'^%s\s*:\s*(#.*)?$' % re.escape(sec), l)]
    if len(hs) != 1: raise ValueError('section %s: %d occurrences' % (sec, len(hs)))
    h = hs[0]; e = h + 1
    while e < len(lines) and (lines[e].strip() == '' or lines[e].startswith((' ', '\t')) or lines[e].lstrip().startswith('#')):
        e += 1
    rx = re.compile(r'^(\s+)%s(\s*:\s*)([^#\s]+)(.*)$' % re.escape(key))
    hit = [i for i in range(h + 1, e) if rx.match(lines[i])]
    if len(hit) > 1: raise ValueError('%s.%s: %d occurrences' % (sec, key, len(hit)))
    if hit:
        m = rx.match(lines[hit[0]]); lines[hit[0]] = m.group(1) + key + m.group(2) + val + m.group(4); how = 'changed'
    else:
        ind = next((re.match(r'^(\s+)\S', lines[i]).group(1) for i in range(h + 1, e) if re.match(r'^\s+[A-Za-z]', lines[i])), '  ')
        lines.insert(h + 1, '%s%s: %s   # p75' % (ind, key, val)); how = 'added'
    return '\n'.join(lines), how
def parse(txt):
    import yaml
    return yaml.safe_load('\n'.join(l for l in txt.split('\n') if not l.startswith('%')))
def make_cfg(v, base, ch):
    if not ch: return BASE[base], []
    txt = open(BASE[base]).read(); log = []
    for sec, kv in ch.items():
        for k, val in kv.items():
            txt, how = set_key(txt, sec, k, val); log.append('%s.%s = %s (%s)' % (sec, k, val, how))
    try:
        d = parse(txt); d0 = parse(open(BASE[base]).read())
        for sec, kv in ch.items():
            for k, val in kv.items():
                assert float(d[sec][k]) == float(val), (sec, k)
        diff = [(s, k) for s in set(d) | set(d0) for k in set(d.get(s) or {}) | set(d0.get(s) or {})
                if (d.get(s) or {}).get(k) != (d0.get(s) or {}).get(k)]
        assert sorted(diff) == sorted((s, k) for s, kv in ch.items() for k in kv), diff
    except ImportError:
        log.append('(PyYAML not available -- parse check skipped)')
    p = O + '/cfg_%s.yml' % v
    if not (os.path.exists(p) and open(p).read() == txt): open(p, 'w').write(txt)
    return p, log

def runs():
    os.makedirs(O, exist_ok=True)
    if os.path.exists(LOCK):
        pid = open(LOCK).read().strip()
        if pid and os.path.exists('/proc/' + pid): sys.exit('!! p75 is already running (pid %s)' % pid)
    open(LOCK, 'w').write(str(os.getpid()))
    try:
        need = [PROBE, INFER, ENG] + list(BASE.values()) + [VID % s for s in SS] + [GT % s for s in SS]
        miss = [p for p in need if not os.path.exists(p)]
        if miss: sys.exit('!! missing:\n  ' + '\n  '.join(miss))
        oldmiss = [OLD % (t, s) for t in BASE for s in SS if not json_ok(OLD % (t, s))]
        if oldmiss: sys.exit('!! p44 logs missing (%d), e.g. %s -- report this' % (len(oldmiss), oldmiss[0]))
        eng0 = sha(ENG)
        CF = {}
        for v, base, ch in VAR:
            CF[v], log = make_cfg(v, base, ch)
            print('  %-9s <- %s %s' % (v, os.path.basename(BASE[base]), '; '.join(log) if log else '(unchanged)'), flush=True)
        todo = [(v, s) for v, _, _ in VAR for s in seqs(v) if not json_ok(NEW % (v, s))]
        print('DeepStream: %d runs remaining' % len(todo), flush=True); t_loop = time.time(); failed = set()
        for i, (v, s) in enumerate(todo, 1):
            if v in failed: continue
            t0 = time.time()
            try:
                r = subprocess.run(['python3', PROBE, VID % s, CF[v], NEW % (v, s), INFER], cwd=W, capture_output=True, text=True, timeout=T_RUN); err = tail(r)
            except subprocess.TimeoutExpired:
                err = 'time limit exceeded (%d s)' % T_RUN
            ok = json_ok(NEW % (v, s)); eta = (time.time() - t_loop) / i * (len(todo) - i) / 60
            print('  [%2d/%d] %-9s SSMOT_%-2d %s (%.0f s, about %.0f min left)' % (i, len(todo), v, s, 'ok' if ok else 'ERROR', time.time() - t0, eta), flush=True)
            if not ok:
                open(O + '/greska_%s.txt' % v, 'w').write(err); failed.add(v)
                print('   !! %s fails -- skipping this variant (tail -40 %s/greska_%s.txt)' % (v, O, v), flush=True)
                if v.endswith('_rep'): sys.exit('!! reproducibility check failed -- STOP')
            if v.endswith('_rep') and ok and s == seqs(v)[-1]:
                for s2 in seqs(v):
                    if norm(NEW % (v, s2)) != norm(OLD % (v[:-4], s2)):
                        sys.exit('!! %s on SSMOT_%d is NOT identical to the p44 log -- the environment changed, STOP and report this' % (v, s2))
                print('   reproducibility %s: identical to p44 on SSMOT %s' % (v[:-4], REP_SS), flush=True)
        need2 = 'dcf1_r0' in failed or any(json_ok(NEW % ('dcf1_r0', x)) and norm(NEW % ('dcf1_r0', x)) != norm(OLD % ('dcf1', x)) for x in SS)
        if need2:
            print('dcf1_r0 %s -> additional pair without re-association (plan, P2b)' % ('fails' if 'dcf1_r0' in failed else 'differs from NvDCF r1'), flush=True)
            for v, base, ch in VAR2:
                CF[v], log = make_cfg(v, base, ch); print('  %-9s <- %s' % (v, '; '.join(log)), flush=True)
            for v, s in [(v, s) for v, _, _ in VAR2 for s in SS if not json_ok(NEW % (v, s))]:
                try:
                    r = subprocess.run(['python3', PROBE, VID % s, CF[v], NEW % (v, s), INFER], cwd=W, capture_output=True, text=True, timeout=T_RUN); err = tail(r)
                except subprocess.TimeoutExpired:
                    err = 'time limit exceeded (%d s)' % T_RUN
                ok = json_ok(NEW % (v, s)); print('  %-9s SSMOT_%-2d %s' % (v, s, 'ok' if ok else 'ERROR'), flush=True)
                if not ok:
                    open(O + '/greska_%s.txt' % v, 'w').write(err); break
        print('detector engine: %s' % ('unchanged' if sha(ENG) == eng0 else '!! CHANGED -- report this'), flush=True)
    finally:
        if os.path.exists(LOCK): os.remove(LOCK)

def norm(p):
    return sorted((int(fd['frame']), int(x['id']), tuple(round(float(c), 2) for c in x['box'][:4]))
                  for fd in json.load(open(p)) for x in (fd.get('detections') or []))

# ---------------- evaluation (TrackEval as in p44)
def ocjena():
    try:
        import trackeval  # noqa
        from scipy.stats import wilcoxon
    except Exception as e:
        sys.exit('!! missing package (%s). First run:\n   %s' % (e, REINSTALL))
    import shutil
    H = {}; exec(open(W + '/id_smooth.py').read().split('CFG={')[0].split('"""')[2], H)
    load_tr, load_gt, pick = H['load_tr'], H['load_gt'], H['pick']
    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update(PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False, TIME_PROGRESS=False,
              USE_PARALLEL=False, DISPLAY_LESS_PROGRESS=True)
    rows = lambda d: sorted((int(f), int(o['id']), *map(float, o['box'][:4])) for f, Lst in d.items() for o in Lst)
    def te(files, tag):
        base = O + '/te_' + tag; shutil.rmtree(base, ignore_errors=True); si = {}
        for s, p in files.items():
            tr = load_tr(p); g = load_gt(GT % s, pick(GT % s, tr))
            G = rows(g); R_ = rows(tr); seq = 'S%02d' % s; si[seq] = max([r[0] for r in G + R_] + [1])
            os.makedirs('%s/gt/%s/gt' % (base, seq), exist_ok=True); os.makedirs(base + '/tr/t/data', exist_ok=True)
            with open('%s/gt/%s/gt/gt.txt' % (base, seq), 'w') as f:
                for fr, i, x, y, w, h in G: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,1,1\n' % (fr, i, x, y, w, h))
            with open('%s/tr/t/data/%s.txt' % (base, seq), 'w') as f:
                for fr, i, x, y, w, h in R_: f.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n' % (fr, i, x, y, w, h))
        dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dc.update(GT_FOLDER=base + '/gt', TRACKERS_FOLDER=base + '/tr', BENCHMARK='b', SPLIT_TO_EVAL='', SEQ_INFO=si, TRACKERS_TO_EVAL=['t'],
                  DO_PREPROC=False, SKIP_SPLIT_FOL=True, GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt', TRACKER_SUB_FOLDER='data', PRINT_CONFIG=False)
        M = [trackeval.metrics.HOTA({'PRINT_CONFIG': False}), trackeval.metrics.CLEAR({'PRINT_CONFIG': False}),
             trackeval.metrics.Identity({'PRINT_CONFIG': False})]
        out, _ = trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)], M)
        Rt = out['MotChallenge2DBox']['t']; shutil.rmtree(base, ignore_errors=True); res = {}
        for seq in si:
            h = Rt[seq]['pedestrian']
            res[int(seq[1:])] = {'HOTA': float(np.mean(h['HOTA']['HOTA'])), 'DetA': float(np.mean(h['HOTA']['DetA'])),
                                 'AssA': float(np.mean(h['HOTA']['AssA'])), 'IDF1': float(h['Identity']['IDF1']), 'IDSW': float(h['CLEAR']['IDSW'])}
        return res
    def holm(ps):
        o = np.argsort(ps); hp = [0.0] * len(ps); run = 0.0
        for r, i in enumerate(o):
            run = max(run, min(1.0, (len(ps) - r) * ps[i])); hp[i] = run
        return hp
    def wp(d):
        d = np.asarray(d, float)
        if not np.any(d != 0): return None          # identical outputs: the test is not defined
        return float(wilcoxon(d).pvalue)
    fp = lambda p: 'identical' if p is None else ('< 0.001' if p < 0.001 else '%.3f' % p)
    sets = {t: {s: OLD % (t, s) for s in SS} for t in BASE}
    sets.update({v: {s: NEW % (v, s) for s in SS} for v, _, _ in VAR + VAR2 if not v.endswith('_rep')})
    R = {}
    P('p75 -- Re-ID on/off within the same configuration (16 SSMOT, YOLO11m, streammux 1920x1080 as in p44; TrackEval)')
    for k, files in sets.items():
        have = [s for s, p in files.items() if json_ok(p)]
        if len(have) < len(SS):
            if k not in ('dcf1_ra0', 'dcf1_r0b') or have: P('  (%s: %d/%d logs -- skipping)' % (k, len(have), len(SS)))
            continue
        R[k] = te(files, k)
    P(); P('  %-42s %7s %7s %7s %7s %8s' % ('', 'HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW/seq'))
    mean = lambda k, m: float(np.mean([R[k][s][m] for s in SS]))
    for k in R:
        P('  %-42s %7.3f %7.3f %7.3f %7.3f %8.1f' % (NAME[k], mean(k, 'HOTA'), mean(k, 'DetA'), mean(k, 'AssA'), mean(k, 'IDF1'), mean(k, 'IDSW')))
    d = lambda a, b, m: [R[a][s][m] - R[b][s][m] for s in SS]
    ident = lambda a, b: sum(1 for s in SS if norm(sets[a][s]) == norm(sets[b][s]))
    J = {'R': {k: {str(s): v for s, v in r.items()} for k, r in R.items()}}
    P(); P('=== P1 (main): NvDeepSORT with the Re-ID term switched off (deep_w0)')
    if {'deep', 'deep_w0', 'dcf1', 'dcf2', 'nvsort'} <= set(R):
        P('  output identical to NvDeepSORT on %d/16 sequences' % ident('deep_w0', 'deep'))
        ex = mean('deep', 'IDSW') - mean('dcf1', 'IDSW'); ud = (mean('deep', 'IDSW') - mean('deep_w0', 'IDSW')) / ex if ex else float('nan')
        P('  share of the excess removed (relative to NvDCF r1): %.0f %%' % (100 * ud))
        refs = ['deep', 'dcf1', 'dcf2']; out = {}
        for m in ['IDSW', 'HOTA', 'AssA', 'IDF1', 'DetA']:
            ps = [wp(d('deep_w0', r, m)) for r in refs]; hs = holm([1.0 if p is None else p for p in ps])
            for r, p, h in zip(refs, ps, hs):
                dd = np.array(d('deep_w0', r, m))
                P('  %-4s deep_w0 - %-11s %+8.3f  (higher on %2d, lower on %2d)  p = %s  Holm %s' % (m, NAME[r], dd.mean(), (dd > 0).sum(), (dd < 0).sum(), fp(p), fp(None if p is None else h)))
                out[(m, r)] = (float(dd.mean()), None if p is None else h)
        I = np.array([[R[k][s]['IDSW'] for k in ('deep_w0', 'nvsort', 'dcf1', 'dcf2')] for s in SS])
        P('  deep_w0 has the most IDSW of (deep_w0, NvSORT, NvDCF r1, NvDCF r2) on %d/16 sequences' % int((I[:, 0] > I[:, 1:].max(1)).sum()))
        sig = all(out[('IDSW', r)][0] > 0 and out[('IDSW', r)][1] is not None and out[('IDSW', r)][1] < 0.05 for r in ('dcf1', 'dcf2'))
        ish = ('the excess does NOT come from the Re-ID term' if ud < 0.25 and sig else 'the Re-ID term is a NECESSARY ingredient of the excess' if ud >= 0.5 else 'partial (descriptive)')
        P('  -> OUTCOME P1: %s' % ish); J['P1'] = {'share': ud, 'outcome': ish}
    P(); P('=== P2: does NvDCF r1 use Re-ID in association? (dcf1_r0 vs dcf1)')
    if {'dcf1', 'dcf1_r0'} <= set(R):
        n = ident('dcf1_r0', 'dcf1'); P('  bit-identical output on %d/16 sequences' % n)
        for m in ['IDSW', 'HOTA', 'AssA', 'IDF1']:
            dd = np.array(d('dcf1_r0', 'dcf1', m)); P('  %-4s dcf1_r0 - NvDCF r1 %+8.3f  p = %s' % (m, dd.mean(), fp(wp(dd))))
        ish = 'NvDCF r1 does NOT use Re-ID in association' if n == 16 else 'NvDCF r1 uses Re-ID (output differs on %d/16)' % (16 - n)
        P('  -> OUTCOME P2: %s' % ish); J['P2'] = {'identical': n, 'outcome': ish}
    else: P('  (no data -- see greska_dcf1_r0.txt)')
    if {'dcf1_ra0', 'dcf1_r0b'} <= set(R):
        n = ident('dcf1_r0b', 'dcf1_ra0'); P('  P2b, both without re-association: reidType 0 vs 1 bit-identical on %d/16' % n)
        for m in ['IDSW', 'HOTA', 'AssA', 'IDF1']:
            dd = np.array(d('dcf1_r0b', 'dcf1_ra0', m)); P('  %-4s r0b - ra0 %+8.3f  p = %s' % (m, dd.mean(), fp(wp(dd))))
        dd = np.array(d('dcf1_ra0', 'dcf1', 'IDSW')); P('  (re-association off in NvDCF r1: IDSW %+.2f, p = %s, identical on %d/16)' % (dd.mean(), fp(wp(dd)), ident('dcf1_ra0', 'dcf1')))
        ish = 'NvDCF r1 does NOT use Re-ID in association' if n == 16 else 'NvDCF r1 uses Re-ID in association (differs on %d/16)' % (16 - n)
        P('  -> OUTCOME P2b: %s' % ish); J['P2b'] = {'identical': n, 'outcome': ish}
    P(); P('=== P3: Re-ID in the NvDCF association cost (dcf1_w vs dcf1)')
    if {'dcf1', 'dcf1_w'} <= set(R):
        P('  identical output on %d/16 sequences' % ident('dcf1_w', 'dcf1'))
        for m in ['IDSW', 'HOTA', 'AssA', 'IDF1']:
            dd = np.array(d('dcf1_w', 'dcf1', m)); P('  %-4s dcf1_w - NvDCF r1 %+8.3f  (higher on %2d, lower on %2d)  p = %s' % (m, dd.mean(), (dd > 0).sum(), (dd < 0).sum(), fp(wp(dd))))
        p = wp(d('dcf1_w', 'dcf1', 'IDSW')); up = np.mean(d('dcf1_w', 'dcf1', 'IDSW')) > 0 and p is not None and p < 0.05
        ish = 'Re-ID in the NvDCF association INCREASES the switches' if up else 'Re-ID in the NvDCF association does not significantly increase the switches'
        P('  -> OUTCOME P3: %s' % ish); J['P3'] = {'outcome': ish}
    P(); P('=== P4 (descriptive): NvDCF r2 without Re-ID in re-association (dcf2_r0 vs dcf2; the visual tracker is not bit-reproducible)')
    if {'dcf2', 'dcf2_r0'} <= set(R):
        for m in ['IDSW', 'HOTA', 'AssA', 'IDF1']:
            dd = np.array(d('dcf2_r0', 'dcf2', m)); P('  %-4s dcf2_r0 - NvDCF r2 %+8.3f  p = %s' % (m, dd.mean(), fp(wp(dd))))
    json.dump(J, open(O + '/p75_rezultati.json', 'w'), indent=1)
    open(OUT, 'w').write('\n'.join(L) + '\n'); print('\nresult: %s' % OUT)

if __name__ == '__main__':
    if '--eval' not in sys.argv: runs()
    ocjena()
