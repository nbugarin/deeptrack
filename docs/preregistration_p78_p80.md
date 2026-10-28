# Preregistration of p78, p79 and p80

Written on 28 October 2026 and committed before any of the three analyses was run.
The commit that adds this file, `experiments/p78_kratka_potvrda.py`, `experiments/p79_neinferiornost.py` and `experiments/p80_pokazatelji_detektora.py` is the time stamp of the plan.

## Purpose

These three checks settle the wording of three hypotheses of the doctoral topic before the topic application is submitted:

- p78 decides H3;
- p79 decides H4;
- p80 decides H5 and the condition on the detector in the main hypothesis.

Nothing in the tracker is tuned. All rule values of the tracker library were fixed on 12 October 2026 on five own recordings:

- class penalty 0.5;
- confidence threshold 0.8 with confirmation after 2 matches;
- confirmation 0.6667 s and track lifetime 3.3333 s otherwise.

## Common threshold

A HOTA difference of **δ = 0.005** is treated as negligible. The same value was used in the decision rules of p52 and p69. It is used here both for "no less accurate" (p78) and for non-inferiority (p79).

Consequently, an improvement claimed in the main hypothesis must be larger than δ. Our tracker improves on NvDCF (reidType 1) by +0.009 HOTA (p55), which is larger than δ.

## Hypotheses concerned (working wording)

- **Main hypothesis.**
  - Differences in identity stability between tracker configurations that share the same detector and the same appearance model arise in track management and association, not in the appearance model.
  - When the detector provides informative uncertainty indicators (as defined for H5), track-management rules that use detection confidence and class stability directly:
    - increase tracking accuracy over the shipped configurations with an appearance model;
    - are not less accurate than the best research trackers on the same detections by more than δ;
    - use rule values set outside the test sets.
- **H3.** Confirmation tied to detection confidence (short for confident detections, longer otherwise):
  - is more accurate than the same long confirmation for all tracks;
  - gives fewer identity switches and false tracks than the same short confirmation for all tracks.
- **H4.** The rules of H2 and H3 in the production system:
  - process video faster than the source frame rate on the target hardware;
  - on the same detections are not less accurate than the best research trackers by more than δ.
- **H5.** The advantage carries over to new scenes and cameras without retuning when the detector provides informative uncertainty indicators. Otherwise the advantage is absent. A detector is informative if either:
  - the class of the same vessel changes in at least 1 % of consecutive-frame pairs; or
  - at least 50 % of detections reach the confirmation threshold 0.8.

## p78 — confidence-gated versus fixed short confirmation (H3), GPU

**Setup.** As in p55 and p69:
- 16 SSMOT sequences (1–18 without 7 and 13);
- YOLO11m TensorRT FP16;
- stream multiplexer at 1280×720;
- `MYTRACKER_FPS` set to the frame rate of the sequence.

**New variants.** Only `confHigh` changes. With `confHigh = 0.0` every detection counts as confident, so every track is confirmed after `minHitsHigh = 2` matches.

| Variant | Definition |
|---|---|
| `k2` | p55 `prob` (gate + seconds, class gate as the baseline) with `confHigh = 0.0` |
| `k2k` | p55 `final` (gate + seconds + class penalty 0.5) with `confHigh = 0.0` |

The existing p55 logs `base`, `prob` and `final` are reused.

**Reproducibility check.** `final` is rerun on SSMOT 1 and must be identical to `log_p55_final_ss_1.json`; otherwise stop.

**Metrics.** TrackEval HOTA, DetA, AssA, IDF1, IDSW, FP, FN and Frag. Two-sided Wilcoxon signed-rank test, paired by sequence, n = 16.

**Primary contrast.** `final − k2k`. The decision rule is applied in this order:

| Outcome | Condition | Consequence |
|---|---|---|
| **A+** | mean ΔHOTA > +δ and Wilcoxon p(HOTA) < 0.05 | the gate is more accurate than short confirmation for all; H3 holds |
| **A** | mean ΔHOTA ≥ −δ, and `final` has fewer FP or fewer IDSW than `k2k` with Holm-adjusted p < 0.05 (Holm over FP and IDSW) | H3 holds as worded |
| **B** | mean ΔHOTA < −δ and Wilcoxon p(HOTA) < 0.05 | short confirmation for all is more accurate; H3 is reworded: confirmation length matters, the confidence gate has no advantage |
| **C** | otherwise | the gate has no measurable effect of its own beside short confirmation; H3 is reworded as short confirmation and the gate is described only descriptively |

**Secondary contrasts** (descriptive, uncorrected): `prob − k2`, `k2 − base` and `prob − base`.

## p79 — non-inferiority and throughput (H4), CPU, existing results

### (a) Non-inferiority on the same detections (p56)

- **Data.** `scpilot/p56/p56_sve.json`, per-sequence HOTA, 16 SSMOT sequences.
- **Difference.** For every research tracker X from BoxMOT present in the file (ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, StrongSORT, Hybrid-SORT, BoostTrack, ImprAssoc): d = HOTA(ours, final) − HOTA(X) per sequence.
- **Test.** One-sided Wilcoxon signed-rank test of d + δ > 0, with Holm correction over the research trackers.
- **Descriptive.** 90 % CI of the mean d: paired t, and bootstrap percentile (10,000 resamples, seed 0).
- **Rule.** The accuracy part of H4 holds if non-inferiority is shown (Holm < 0.05) against every research tracker. Otherwise H4 is stated as "no significant difference", and the trackers against which non-inferiority is not shown are named.

### (b) Throughput without start-up

- **Data.** Per-sequence (frames, seconds) from `stanje.json` (`vrijeme`):

  | Run | Data set | Resolution, source frame rate |
  |---|---|---|
  | `p64_ssmot`, `p64_coco` | SMD-Plus | 1920×1080, 30 FPS |
  | `p73_ssmot`, `p73_coco` | ShipsMOT | 1920×1080, 30 FPS |
  | `p60_coco` | FVessel | 2560×1440, 25 FPS |

- **Throughput.** 1 / slope of the least-squares line seconds ~ frames over the sequences, for each tracker.
- **Rule.** A set runs in real time if the throughput of our tracker is at least the source frame rate. Where it is not, the real-time claim of H4 is restricted to the sets where it holds, and the limitation is stated.

## p80 — detector uncertainty indicators (H5), CPU, existing logs

**Indicators.** Computed from the output logs of our tracker (final variant), over all output boxes:

| Indicator | Definition |
|---|---|
| conf share | fraction of boxes whose detection confidence is ≥ 0.8 |
| class change rate | fraction of pairs (same track ID, frames f and f+1) whose class differs |

**Thresholds** (fixed now). A detector is **informative** if the class change rate is ≥ 1 % or the conf share is ≥ 50 %.
- 50 % is a simple majority.
- 1 % lies between 0.45 % and 4.37 %:
  - 0.45 % is SSMOT, where disabling the NVIDIA class check had no effect (p66);
  - 4.37 % is SMD-Plus, where the effect was strong (p65).
- These thresholds were chosen after those results (post hoc), but before the ShipsMOT indicators were computed.

**Sets.** SSMOT with the SSMOT detector (p55), SMD-Plus with both detectors (p64), FVessel with the COCO detector (p60), and the 14 fixed-camera ShipsMOT sequences with both detectors (p73).

**Predictions for ShipsMOT.** The tracking outcomes are already known; the indicators are not yet computed.
- **SSMOT detector** (ours had the highest HOTA): informative, i.e. at least one indicator above its threshold.
- **COCO detector** (NvDCF reidType 1 significantly better): not informative, i.e. class change rate < 1 % and conf share < 50 %.

**Consequences.**
- If the COCO conf share on ShipsMOT is ≥ 50 %, the confidence branch of H5 is contradicted. H5 and the detector condition in the main hypothesis are then restricted to the class branch.
- If the SSMOT detector is not informative on ShipsMOT, H5 is contradicted and is stated descriptively.
- If any of the four already-known sets falls on the other side of a threshold when recomputed with this code, this is reported.
