# Experiments and where they appear in the article

Section and table numbers refer to the version submitted to IEEE Access.

| Script | What it does | Article |
|---|---|---|
| `p44_mdpi_11m.py` | Exports YOLO11m for DeepStream, builds the TensorRT engine, runs the four NVIDIA trackers and the end-to-end Ultralytics ByteTrack and BoT-SORT | Ultralytics rows of Table V; Section V-B |
| `p52_vlastiti_11m.py` | Baseline and final variant of the library with YOLO11m (stream multiplexer at 1920×1080) | Section V-F (1920×1080 comparison) |
| `p53_rastav_11m.py` | Decomposition of the two rules and sensitivity at 1920×1080 | Section V-F (decomposition repeated at 1920×1080) |
| `p54_detekcije.py` | Detection quality of the same YOLO11m weights in DeepStream and Ultralytics, without a tracker; exports the DeepStream detections | Table X; detections for `p56` |
| `p55_izvorna_rezolucija.py` | All DeepStream trackers with the stream multiplexer at the native 1280×720; decomposition, sensitivity, identity-switch diagnosis, throughput | Tables VII, VIII, IX, XI; Figs. 4 and 5 |
| `p56_iste_detekcije.py` | Seven BoxMOT trackers on the replayed DeepStream detections; comparison of all twelve trackers with Holm-corrected Wilcoxon tests | Tables V and VI; Figs. 2 and 3; BoxMOT throughput in Table XI |
| `p60_coco_priprema.py` | Prepares the COCO-pretrained YOLO11m (class boat) for DeepStream | Section V-H |
| `p60_detektor.py` | Quick recall check of both detectors on FVessel (Ultralytics) | Section V-H |
| `p60_fvessel.py` | Trackers on FVessel with the COCO detector; `--pomak=K` repeats the evaluation with a fixed frame offset | Table XII (FVessel); offset sensitivity in Section V-H |
| `p62_smd_provjera.py` | Checks the SMD-Plus annotation format and videos | Section V-H |
| `p63_smd_gt.py` | Derives track identities for SMD-Plus from the per-frame annotation | Section V-H |
| `p64_smd.py` | Trackers on 40 SMD-Plus sequences with both detectors | Table XII (SMD-Plus) |
| `p65_klasa.py` | NVIDIA trackers with class-agnostic matching (`checkClassMatch: 0`) on SMD-Plus | Table XIV |
| `p66_klasa_ssmot.py` | The same on the 16 SSMOT sequences | Table XIV |
| `p67_kalibracija.py` | Detections of both detectors in the DeepStream pipeline without a tracker on SSMOT, SMD-Plus and FVessel; recall and precision on SMD-Plus. The script also contains the confidence-calibration analysis of a companion study | Detector recall and precision in Section V-H |
| `p69_rastav_probacije.py` | Separates the two parts of the probation: the confidence gate with durations in frames, and the durations in seconds without the gate | Section V-C (decomposition of the probation) |
| `p70_rukovatelj.py` | Reruns OC-SORT, Deep OC-SORT and Hybrid-SORT as in `p56` and counts how often the exception handler in the shared Kalman filter is invoked; checks that the outputs equal those of `p56` | Sections IV-D and VII |

For the experiments that test a hypothesis (`p52`–`p56`, `p60_fvessel`, `p64`–`p67`, `p69`, `p70`), the comparisons, metrics, tests and decision rules were written down before the run and are quoted at the top of the script. The other scripts prepare data or detectors.
