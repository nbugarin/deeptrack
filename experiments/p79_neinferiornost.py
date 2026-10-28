#!/usr/bin/env python3
"""p79: NON-INFERIORITY AGAINST THE RESEARCH TRACKERS AND THROUGHPUT WITHOUT START-UP (H4). CPU only, existing results.

Plan and decision rules written BEFORE the analysis: docs/preregistration_p78_p80.md (committed before any run).
(a) Non-inferiority, same detections (p56): d_s = HOTA(ours, final) - HOTA(X) per SSMOT sequence (n = 16) for every
    research tracker X from BoxMOT. One-sided Wilcoxon signed-rank test of d_s + delta > 0 (delta = 0.005 HOTA),
    Holm over the trackers. Descriptive: 90 % CI of the mean difference (paired t and bootstrap percentile, 10,000
    resamples, seed 0). Rule: the accuracy part of H4 holds if non-inferiority is shown (Holm < 0.05) against every
    research tracker; otherwise H4 is stated as "no significant difference" and the trackers concerned are named.
(b) Throughput: per-sequence (frames, seconds) from stanje.json ('vrijeme') of p64_ssmot, p64_coco (SMD-Plus,
    1920x1080, 30 FPS), p73_ssmot, p73_coco (ShipsMOT, 1920x1080, 30 FPS) and p60_coco (FVessel, 2560x1440, 25 FPS).
    Throughput = 1 / slope of the least-squares line seconds ~ frames (excludes the start-up). Rule: real time on a
    set if the throughput of ours >= the source FPS; if not on 1080p, the real-time claim of H4 is restricted to the
    sets where it holds and the limitation is stated.
Run:     python3 /workspace/p79_neinferiornost.py
Output:  /workspace/scpilot/p79_neinferiornost.txt (+ scpilot/p79_sve.json)
"""
import os, sys, json
import numpy as np
W = os.environ.get('P79_W', '/workspace'); S = W + '/scpilot'
DELTA = 0.005
P56 = S + '/p56/p56_sve.json'
ME = 'ours, final'
RESEARCH = ['ByteTrack', 'BoT-SORT', 'OC-SORT', 'Deep OC-SORT', 'StrongSORT', 'Hybrid-SORT', 'BoostTrack', 'ImprAssoc']
SPEED = [('SMD-Plus, SSMOT detector', S + '/p64_ssmot/stanje.json', 30.0), ('SMD-Plus, COCO detector', S + '/p64_coco/stanje.json', 30.0),
         ('ShipsMOT, SSMOT detector', S + '/p73_ssmot/stanje.json', 30.0), ('ShipsMOT, COCO detector', S + '/p73_coco/stanje.json', 30.0),
         ('FVessel, COCO detector', S + '/p60_coco/stanje.json', 25.0)]
L = []
def P(x=''): print(x, flush=True); L.append(x)
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); run = 0
    for r, i in enumerate(o): run = max(run, min(1, ps[i] * (len(ps) - r))); out[i] = run
    return out
def noninf_p(d):
    """One-sided Wilcoxon signed-rank test: H0 median(d + delta) <= 0, H1 > 0."""
    from scipy.stats import wilcoxon
    x = np.asarray(d, float) + DELTA
    if not np.any(x != 0): return 1.0
    try: return float(wilcoxon(x, alternative='greater').pvalue)
    except ValueError: return float('nan')
def ci90(d, seed=0, nb=10000):
    from scipy.stats import t as tdist
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); q = tdist.ppf(0.95, n - 1)
    rng = np.random.default_rng(seed); bs = d[rng.integers(0, n, size=(nb, n))].mean(axis=1)
    return (m - q * se, m + q * se), (float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))
def part_a(OUT):
    if not os.path.exists(P56): P('(a) !! missing %s -- p56 evaluation first' % P56); return
    D = json.load(open(P56)); PS = D.get('po_snimkama', {})
    if ME not in PS: P('(a) !! "%s" not in p56_sve.json (keys: %s)' % (ME, ', '.join(PS))); return
    seqs = sorted(PS[ME], key=int)
    refs = [k for k in RESEARCH if k in PS and sorted(PS[k], key=int) == seqs]
    P('=== (a) NON-INFERIORITY on the same detections (p56), %d SSMOT sequences, delta = %.3f HOTA' % (len(seqs), DELTA))
    P('    one-sided Wilcoxon of d + delta > 0, Holm over %d research trackers; 90 %% CI of the mean d (descriptive)' % len(refs))
    rows = []
    for k in refs:
        d = [PS[ME][s]['HOTA'] - PS[k][s]['HOTA'] for s in seqs]
        (tl, th), (bl, bh) = ci90(d)
        rows.append((k, float(np.mean(d)), noninf_p(d), tl, th, bl, bh))
    hs = holm([r[2] for r in rows])
    OUT['a'] = {}
    P('  %-14s %9s %9s %9s %-21s %-21s %s' % ('tracker', 'mean d', 'p', 'Holm', '90% CI (t)', '90% CI (bootstrap)', 'non-inferior'))
    for (k, md, p, tl, th, bl, bh), h in zip(rows, hs):
        ok = h < 0.05
        P('  %-14s %+9.4f %9.4f %9.4f [%+.4f, %+.4f]   [%+.4f, %+.4f]   %s' % (k, md, p, h, tl, th, bl, bh, 'YES' if ok else 'not shown'))
        OUT['a'][k] = {'mean_d': md, 'p': p, 'holm': h, 'ci_t': [tl, th], 'ci_boot': [bl, bh], 'noninferior': ok}
    fail = [k for k in OUT['a'] if not OUT['a'][k]['noninferior']]
    verdict = ('HOLDS: non-inferior (delta %.3f) to every research tracker' % DELTA) if refs and not fail else \
              ('NOT SHOWN against: %s -> H4 stated as "no significant difference", these trackers named' % ', '.join(fail) if refs else 'no research trackers found')
    P('  RULE (a): %s' % verdict); OUT['a_verdict'] = verdict
def part_b(OUT):
    P('\n=== (b) THROUGHPUT WITHOUT START-UP (1 / slope of seconds ~ frames over the sequences)')
    OUT['b'] = {}
    for lab, path, src in SPEED:
        if not os.path.exists(path): P('  %-26s missing %s' % (lab, path)); continue
        V = json.load(open(path)).get('vrijeme', {})
        by = {}
        for key, nv in V.items():
            k, _, _ = key.partition('|'); by.setdefault(k, []).append((float(nv[0]), float(nv[1])))
        res = {}
        for k, pts in sorted(by.items()):
            if len(pts) < 5: continue
            n = np.array([p[0] for p in pts]); t = np.array([p[1] for p in pts])
            slope, icpt = np.polyfit(n, t, 1)
            res[k] = {'throughput': float(1 / slope) if slope > 0 else float('nan'), 'startup_s': float(icpt),
                      'fps_with_startup': float(n.sum() / t.sum()), 'n_seq': len(pts)}
        if 'final' not in res: P('  %-26s no timings of ours' % lab); continue
        own = res['final']['throughput']; rt = own >= src
        P('  %-26s source %4.0f FPS | ours %5.1f FPS (with start-up %5.1f; start-up %.1f s) -> %s' % (
            lab, src, own, res['final']['fps_with_startup'], res['final']['startup_s'], 'REAL TIME' if rt else 'BELOW real time'))
        P('  %-26s NVIDIA: %s' % ('', ', '.join('%s %.1f' % (k.replace('nv_', ''), v['throughput']) for k, v in res.items() if k != 'final')))
        OUT['b'][lab] = {'source_fps': src, 'trackers': res, 'real_time': bool(rt)}
    below = [k for k, v in OUT['b'].items() if not v['real_time']]
    verdict = 'real time on all measured sets' if OUT['b'] and not below else ('BELOW real time on: %s -> real-time claim restricted, limitation stated' % ', '.join(below) if OUT['b'] else 'no timings found')
    P('  RULE (b): %s' % verdict); OUT['b_verdict'] = verdict
if __name__ == '__main__':
    OUT = {'delta': DELTA}
    P('=== p79: H4 -- non-inferiority and throughput (rules: docs/preregistration_p78_p80.md)')
    part_a(OUT); part_b(OUT)
    os.makedirs(S, exist_ok=True)
    open(S + '/p79_neinferiornost.txt', 'w').write('\n'.join(L) + '\n'); json.dump(OUT, open(S + '/p79_sve.json', 'w'), indent=1)
    print('\nOutput: %s/p79_neinferiornost.txt' % S, flush=True)
