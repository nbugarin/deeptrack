# Class-tolerant association with confidence-gated probation for vessel tracking in NVIDIA DeepStream

Code accompanying the article

> N. Bugarin, M. Miličević, I. Marin, and S. Gotovac, "Class-Tolerant Association With Confidence-Gated Probation for Vessel Tracking in NVIDIA DeepStream," submitted to *IEEE Access*, 2026.

The repository contains an association library for the NVIDIA DeepStream low-level tracker interface (NvMOT) and the scripts used to run and evaluate every experiment in the article. The library adds two rules for maritime video to a SORT-type tracker with an appearance branch:

- **Class-tolerant cost.** A class mismatch between a track and a detection adds a penalty λ to the association cost instead of forbidding the match.
- **Confidence-gated probation.** A track whose detection has confidence ≥ τ is confirmed after two matches; any other track is confirmed after a fixed time T<sub>p</sub> in seconds. Track lifetime T<sub>a</sub> is also given in seconds.

## Repository layout

| Folder | Content |
|---|---|
| `tracker/` | C++ source of the association library (`libmoj.so`) and its Makefile |
| `configs/` | DeepStream inference and tracker configuration files used in the article |
| `pipeline/` | Python scripts that run the DeepStream pipeline and write tracker output and detections to JSON |
| `experiments/` | One script per experiment; see [docs/experiments.md](docs/experiments.md) for the mapping to the tables and figures of the article |
| `data/` | How to obtain the datasets; no data are included |

## Requirements

- NVIDIA DeepStream 7.0 with CUDA 12 and TensorRT (the article used a GeForce GTX 1660 SUPER under Ubuntu 22.04)
- Eigen 3 (`libeigen3-dev`)
- Python 3.10 with NumPy, SciPy and OpenCV
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) for HOTA, IDF1 and CLEAR metrics
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) 10.0.83 for the open-source reference trackers
- [Ultralytics](https://github.com/ultralytics/ultralytics) 8.3.0 and [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) for the YOLO11m detector

## Building and using the library

```bash
cd tracker
make            # builds libmoj.so; paths to DeepStream and CUDA are set at the top of the Makefile
```

In the DeepStream tracker configuration, point `ll-lib-file` to the built library. The frame rate of the video is passed in the environment variable `MYTRACKER_FPS`, so that durations given in seconds are converted to frames for each video.

### Parameters

The parameters are set in `tracker/src/track.h`. The experiment scripts change them there and rebuild the library.

| Parameter in the article | Name in `track.h` | Baseline | Final variant |
|---|---|---|---|
| Overlap, size and appearance weights w<sub>o</sub>, w<sub>s</sub>, w<sub>a</sub> | `wReid` (w<sub>a</sub>) | 0.8 | 0.8 |
| Descriptors stored per track G | `galleryMax` | 5 | 5 |
| Maximum association cost c<sub>max</sub> | `maxCost` | 0.75 | 0.75 |
| Inertia discount η | `inertia` | 0.10 | 0.10 |
| Detection suppression threshold (IoU) | `detNms` | 0.70 | 0.70 |
| Kalman measurement noise | `wMeas` | 0.15 | 0.15 |
| Class mismatch | `classGate`, `classPenalty` (λ) | match excluded (`classGate = 1`) | penalty 0.5 (`classGate = 0`) |
| Confirmation | `minHits`, `confHigh` (τ), `minHitsSec` (T<sub>p</sub>) | 20 frames | 2 matches if p ≥ 0.8, otherwise 0.6667 s |
| Track lifetime without a match | `maxAge`, `maxAgeSec` (T<sub>a</sub>) | 100 frames | 3.3333 s |

## Reproducing the experiments

Each script in `experiments/` writes down at the top what it does, what it needs and how it is run. The scripts expect the DeepStream container with the datasets, engines and configuration files under `/workspace`, or in the folder given in the environment variable `P<NN>_W` of each script (for example `P55_W`). Scripts `p44` and `p52`–`p56` can be rerun in evaluation mode only, without a GPU, with `--eval`.

The comments and the output of the scripts are in Croatian.

## Data and models

No data and no model weights are included. See [data/README.md](data/README.md) for how to obtain SSMOT, SMD-Plus and FVessel. The YOLO11m and OSNet weights were trained on the SSMOT detection and Re-ID subsets; they are available from the authors upon reasonable request, with the agreement of the author of SSMOT.

## Use of generative AI

Parts of the experiment and evaluation scripts were written with the assistance of a generative AI system (Claude, Anthropic). The authors reviewed the scripts and are responsible for them.

## License

MIT, see [LICENSE](LICENSE). The NVIDIA DeepStream SDK, TensorRT and the NVIDIA reference tracker configurations are not part of this repository and are subject to their own licenses.
