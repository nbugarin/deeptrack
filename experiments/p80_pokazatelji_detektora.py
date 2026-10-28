#!/usr/bin/env python3
"""p80: DETECTOR UNCERTAINTY INDICATORS PER DATA SET AND DETECTOR (H5). CPU only, existing logs of our tracker (final).

Plan, thresholds and predictions written BEFORE the analysis: docs/preregistration_p78_p80.md (committed before any run).
Indicators, computed over all output boxes of ours (final variant):
  conf share        = fraction of boxes with detection confidence >= 0.8 (the confirmation threshold);
  class change rate = fraction of pairs (same track ID, frames f and f+1) whose class differs.
A detector is "informative" on a set if class change rate >= 1 % or conf share >= 50 %.
Predictions for ShipsMOT, 14 fixed-camera sequences (outcomes known, indicators not yet computed):
  SSMOT detector -> informative;  COCO detector -> class change rate < 1 % and conf share < 50 %.
Run:     python3 /workspace/p80_pokazatelji_detektora.py
Output:  /workspace/scpilot/p80_pokazatelji.txt (+ scpilot/p80_sve.json)
"""
import os, re, sys, json, glob
W = os.environ.get('P80_W', '/workspace'); S = W + '/scpilot'
TAU = 0.8; T_CLASS = 0.01; T_CONF = 0.50
SS16 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
FIX14 = [1, 13, 18, 34, 35, 37, 40, 55, 58, 59, 62, 71, 92, 97]          # p71b, rule set before the measurement
SETS = [('SSMOT, SSMOT detector', [W + '/log_p55_final_ss_%d.json' % s for s in SS16], 'known: advantage (via confidence, H3)'),
        ('SMD-Plus, SSMOT detector', sorted(glob.glob(S + '/p64_ssmot/log_p60_final_*.json')), 'known: advantage (via class, H2)'),
        ('SMD-Plus, COCO detector', sorted(glob.glob(S + '/p64_coco/log_p60_final_*.json')), 'known: no advantage'),
        ('FVessel, COCO detector', sorted(glob.glob(S + '/p60_coco/log_p60_final_*.json')), 'known: equal to the best'),
        ('ShipsMOT fixed, SSMOT detector', [S + '/p73_ssmot/log_p60_final_shipsmot_%03d.json' % n for n in FIX14], 'PREDICTION: informative'),
        ('ShipsMOT fixed, COCO detector', [S + '/p73_coco/log_p60_final_shipsmot_%03d.json' % n for n in FIX14], 'PREDICTION: not informative')]
CK = ('class', 'class_id', 'classId', 'cls', 'label'); PK = ('conf', 'confidence', 'score')
L = []
def P(x=''): print(x, flush=True); L.append(x)
def first(x, keys):
    for k in keys:
        if k in x and x[k] is not None: return x[k]
    return None
def indicators(paths):
    n_box = n_conf = n_hi = n_pair = n_chg = 0; used = 0
    for p in paths:
        if not os.path.exists(p): continue
        try: data = json.load(open(p))
        except Exception: continue
        used += 1; last = {}
        for fd in sorted(data, key=lambda d: int(d['frame'])):
            f = int(fd['frame']); cur = {}
            for x in fd.get('detections') or []:
                n_box += 1; c = first(x, PK)
                if c is not None:
                    n_conf += 1; n_hi += float(c) >= TAU
                k = first(x, CK); tid = x.get('id')
                if k is not None and tid is not None:
                    cur[tid] = k
                    if tid in last and last[tid][0] == f - 1:
                        n_pair += 1; n_chg += last[tid][1] != k
            for tid, k in cur.items(): last[tid] = (f, k)
    return {'files': used, 'boxes': n_box, 'conf_share': (n_hi / n_conf) if n_conf else float('nan'),
            'class_change': (n_chg / n_pair) if n_pair else float('nan'), 'pairs': n_pair, 'boxes_with_conf': n_conf}
if __name__ == '__main__':
    OUT = {}
    P('=== p80: detector uncertainty indicators (rules: docs/preregistration_p78_p80.md)')
    P('    informative if class change rate >= %.0f %% or share of boxes with confidence >= %.1f is >= %.0f %%' % (100 * T_CLASS, TAU, 100 * T_CONF))
    P('  %-32s %5s %9s %11s %12s %-14s %s' % ('set, detector', 'files', 'boxes', 'conf>=0.8', 'class chg', 'informative', 'note'))
    for lab, paths, note in SETS:
        r = indicators(paths)
        if not r['files']: P('  %-32s no logs found' % lab); OUT[lab] = r; continue
        cls_ok = r['class_change'] == r['class_change'] and r['class_change'] >= T_CLASS
        conf_ok = r['conf_share'] == r['conf_share'] and r['conf_share'] >= T_CONF
        r['informative'] = bool(cls_ok or conf_ok); r['why'] = [w for w, ok in (('class', cls_ok), ('confidence', conf_ok)) if ok]
        P('  %-32s %5d %9d %10.1f%% %11.2f%% %-14s %s' % (lab, r['files'], r['boxes'], 100 * r['conf_share'], 100 * r['class_change'],
                                                    ('YES (' + '+'.join(r['why']) + ')') if r['informative'] else 'no', note))
        OUT[lab] = r
    P('\n=== RULE FROM THE PREREGISTRATION (ShipsMOT fixed)')
    a = OUT.get('ShipsMOT fixed, SSMOT detector', {}); b = OUT.get('ShipsMOT fixed, COCO detector', {})
    if a.get('files') and b.get('files'):
        if b.get('conf_share', 0) >= T_CONF:
            v = 'CONTRADICTED (confidence branch): COCO has conf share >= 50 % without an advantage -> H5 and the detector condition in H restricted to the class branch'
        elif not b.get('informative') and a.get('informative'):
            v = 'CONFIRMED: SSMOT detector informative (%s), COCO detector not informative' % '+'.join(a['why'])
        elif not a.get('informative'):
            v = 'CONTRADICTED: SSMOT detector not informative although ours had the highest HOTA -> H5 stated descriptively'
        else:
            v = 'CONTRADICTED (class branch): COCO class change rate >= 1 % without an advantage -> report, H5 reworded'
    else: v = 'logs missing -- cannot decide'
    P('  %s' % v); OUT['verdict'] = v
    known = [('SSMOT, SSMOT detector', True), ('SMD-Plus, SSMOT detector', True), ('SMD-Plus, COCO detector', False), ('FVessel, COCO detector', False)]
    mism = [k for k, exp in known if OUT.get(k, {}).get('files') and OUT[k].get('informative') != exp]
    P('  known sets on the expected side of the thresholds: %s' % ('all' if not mism else 'NOT for ' + ', '.join(mism) + ' -- report this'))
    OUT['known_mismatch'] = mism
    os.makedirs(S, exist_ok=True)
    open(S + '/p80_pokazatelji.txt', 'w').write('\n'.join(L) + '\n'); json.dump(OUT, open(S + '/p80_sve.json', 'w'), indent=1)
    print('\nOutput: %s/p80_pokazatelji.txt' % S, flush=True)
