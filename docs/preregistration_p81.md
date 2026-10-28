# Preregistration of p81

Written on 28 October 2026 and committed before the run.
The commit that adds this file and `experiments/p81_jedna_klasa.py` is the time stamp of the plan.

## Purpose

H5 of the doctoral topic has two parts:

- the advantage of the track-management rules carries over to new scenes and cameras without retuning when the detector provides informative uncertainty indicators;
- when the detector provides neither indicator, the advantage is absent.

A detector is **informative** if the class of the same vessel changes in at least 1 % of consecutive-frame pairs, or if at least 50 % of detections reach the confirmation threshold 0.8 (`docs/preregistration_p78_p80.md`).

So far the second part rests only on comparisons between two different detectors: YOLO11m trained on SSMOT and the general COCO YOLO11m. They differ in more than the indicators, for example in recall, precision and box quality.

p81 takes one detector, removes one indicator from its output and changes nothing else. This tests the second part of H5 directly.

## Set and indicators

The set is SMD-Plus VIS_Onshore (40 sequences) with the SSMOT detector, as in p64. p80 measured the indicators on this set:

| Indicator | Value | Threshold | Informative |
|---|---|---|---|
| class change rate | 4.23 % | ≥ 1 % | yes |
| share of detections with confidence ≥ 0.8 | 26.3 % | ≥ 50 % | no |

The detector is therefore informative only through the class.

If the class is removed, the class change rate becomes 0 % and the confidence share stays at about 26 %. The detector is then **not informative**, and H5 predicts that the advantage of our tracker disappears.

## Setup

The pipeline is the same as p64:

- the run machinery of `p60_fvessel.py`;
- the stream multiplexer at the native 1920×1080;
- the same tracker configuration files;
- `MYTRACKER_FPS` set to the frame rate of the sequence.

The trackers are frozen. Nothing is tuned:

- ours, final variant;
- NvSORT;
- NvDeepSORT;
- NvDCF (reidType 1);
- NvDCF (reidType 2).

**The only change.** A pad probe on the sink pad of `nvtracker` sets `class_id = 0` for every detection before it reaches the tracker. Boxes and confidences are unchanged.

The non-maximum suppression of the detector is not changed. A real single-class detector would also suppress overlapping boxes across classes. This is deliberately not done, so that the trackers receive exactly the same boxes as in p64.

**Reference.** The existing p64 logs in `scpilot/p64_ssmot` serve as the reference.

## Checks (stop if any fails)

1. **Reproducibility.** Our final variant is rerun with the unmodified probe on the first sequence. Its output must be identical to the p64 log (same frames, IDs and boxes).
2. **Manipulation.** Every box in every p81 log must have class 0.
3. **Indicators.** The indicators are recomputed on the p81 logs of our tracker with the definitions of p80. The class change rate must be 0 %. The confidence share is reported and is expected to stay below 50 %.

## Evaluation

The evaluation is the same as for p64:

- every frame is evaluated;
- tracker boxes matched to distractors (Buoy, Other) are removed;
- the frame offset is the one of the main p64 evaluation for the SSMOT detector, read from `scpilot/p64_rezultati.json`;
- metrics: TrackEval HOTA, DetA, AssA, IDF1, IDSW, Frag, FP and FN;
- n = 40 sequences, paired by sequence.

δ = 0.005 HOTA, as in p52, p69, p78 and p79.

## Primary quantity

The advantage is measured against **NvDCF (reidType 1)**. It is the strongest shipped configuration with an appearance model, and it is the comparator under which the advantage was absent with the COCO detector (p73, p80).

For each sequence:

- A = HOTA(ours) − HOTA(NvDCF r1);
- A_multi is A from p64, with class information;
- A_single is A from p81, without class information.

**Advantage present** in p81 means both of the following:

- mean A_single > +δ;
- the two-sided Wilcoxon signed-rank test of ours minus NvDCF r1 has a Holm-adjusted p < 0.05. Holm is applied over the four NVIDIA trackers.

**Shrinkage** is D = A_multi − A_single per sequence, tested with a two-sided Wilcoxon signed-rank test.

## Decision rule

| Outcome | Condition | Consequence |
|---|---|---|
| **A** | In p81 the advantage over NvDCF r1 is not present, **and** mean D > +δ with Wilcoxon p(D) < 0.05 | H5 prediction confirmed with a single detector: removing the class indicator alone removes the advantage. The class branch of H5 is supported by a controlled manipulation. |
| **B** | In p81 the advantage over NvDCF r1 is still present | Contradicted: on SMD-Plus the advantage does not come from the class indicator. The statement "when the detector provides neither indicator, the advantage is absent" is not supported. H5 and the detector condition in the main hypothesis are reworded, and this result is reported. |
| **C** | Otherwise | Inconclusive. Reported descriptively; the status of H5 does not change. |

## Secondary contrasts (descriptive, uncorrected)

1. **The same rule against the other NVIDIA trackers** (NvDCF r2, NvSORT, NvDeepSORT). NvDeepSORT is expected to stay worse than ours; it was worse on every set and with both detectors.
2. **Ours in p81 minus ours in p64: the "class penalty 0" control for H2.**
   - In our tracker with `classGate = 0`, the class enters only the class penalty. Suppression of overlapping detections ignores the class.
   - With a single class the penalty never applies, so ours in p81 is our tracker with `classPenalty = 0` on the set where the class changes most.
   - Measures: HOTA, AssA, IDF1, IDSW and Frag.
   - H2 compares the penalty with the hard class gate, so this contrast does not decide H2. It shows whether the penalty adds anything beyond removing the gate.
3. **NVIDIA trackers in p81 against the same trackers with `checkClassMatch: 0` (p65).** This shows whether removing the class at the detector has the same effect as switching off the class check in the tracker.
4. **NVIDIA trackers in p81 against p64.** This is the effect of removing the class on each NVIDIA tracker.

## Expectation stated before the run

p65 ran the NVIDIA trackers with `checkClassMatch: 0` on this set. NvDCF r1 reached HOTA 0.382, against 0.372 for ours. Outcome A is therefore expected.

The value of p81 is in two things:

- it tests the H5 claim in its own terms: the detector loses the indicator and the trackers stay unchanged;
- it provides the class-penalty-0 control for our tracker.
