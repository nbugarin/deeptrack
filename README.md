# Class-tolerant association with confidence-gated probation for vessel tracking in NVIDIA DeepStream

Code accompanying the article

> N. Bugarin, M. Miličević, I. Marin, and S. Gotovac, "Class-Tolerant Association With Confidence-Gated Probation for Vessel Tracking in NVIDIA DeepStream," submitted to *IEEE Access*, 2026.

The repository contains an association library for the NVIDIA DeepStream low-level tracker interface (NvMOT) and the scripts used to run and evaluate every experiment in the article. The library adds two rules for maritime video to a SORT-type tracker with an appearance branch:

- **Class-tolerant cost.** A class mismatch between a track and a detection adds a penalty λ to the association cost instead of forbidding the match.
- **Confidence-gated probation.** A track whose detection has confidence ≥ τ is confirmed after two matches; any other track is confirmed after a fixed time T<sub>p</sub> in seconds. The track lifetime T<sub>a</sub> is also given in seconds.

## Repository layout

| Folder | Content |
|---|---|
| `tracker/` | C++ source of the association library (`libmoj.so`), its Makefile, a test of the two rules (`tests/test_rules.cpp`) and a test of the appearance engine (`test_reid.cpp`) |
| `pipeline/` | Python scripts that run the DeepStream pipeline and write the tracker output to JSON |
| `configs/` | `nvinfer` configurations of the two detectors and the settings of the four NVIDIA trackers |
| `experiments/` | One script per experiment; [docs/experiments.md](docs/experiments.md) maps them to the tables and figures of the article |
| `data/` | How to obtain the datasets; no data are included |

## Requirements

- NVIDIA DeepStream 7.0 with CUDA 12 and TensorRT (the article used a GeForce GTX 1660 SUPER under Ubuntu 22.04)
- Eigen 3 (`libeigen3-dev`)
- Python 3.10 with NumPy, SciPy, OpenCV and the DeepStream Python bindings (`pyds`)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) for HOTA, IDF1 and CLEAR metrics
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) 10.0.83 for the open-source reference trackers
- [Ultralytics](https://github.com/ultralytics/ultralytics) 8.3.0 and [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) for the YOLO11m detector

## Building and using the library

```bash
cd tracker
make            # builds libmoj.so; the paths to DeepStream and CUDA are set at the top of the Makefile
```

In the DeepStream tracker element, set `ll-lib-file` to the built library (see `pipeline/probe_moj_p55.py`). The library reads the frame rate of the video from the environment variable `MYTRACKER_FPS` (default 30), so that durations in seconds are converted to frames for each video. The TensorRT engine of the appearance model is loaded from `/workspace/osnet_vessel.engine` (`tracker/src/tracker_api.cpp`); without it the library runs without the appearance term.

The scripts expect the library in `/workspace/mytracker`, that is, the content of `tracker/` copied there.

### Checking the two rules without DeepStream

```bash
cd tracker
g++ -std=c++14 -O2 -Isrc -I/usr/include/eigen3 tests/test_rules.cpp src/tracker.cpp src/hungarian.cpp src/kalman.cpp -o test_rules
MYTRACKER_FPS=25 ./test_rules
```

The test checks that a class change opens a new track in the baseline and keeps the identifier with the penalty, and that at 25 FPS a confident track is confirmed at its second frame, an uncertain one at frame 17 and a baseline track at frame 20.

### Parameters

All parameters are in `tracker/src/track.h`. The file is in the baseline configuration; the experiment scripts set the values of the final variant (`FINAL` in `experiments/p52_vlastiti_11m.py`), rebuild the library and restore the file afterwards.

| Parameter in the article | Name in `track.h` | Baseline | Final variant |
|---|---|---|---|
| Overlap, size and appearance weights w<sub>o</sub>, w<sub>s</sub>, w<sub>a</sub> | `wIou`, `wSize`, `wReid` | 1.0, 0.3, 0.8 | 1.0, 0.3, 0.8 |
| Descriptors stored per track G | `galleryMax` | 5 | 5 |
| Maximum association cost c<sub>max</sub> | `maxCost` | 0.75 | 0.75 |
| Inertia discount η | `inertia` | 0.10 | 0.10 |
| Detection suppression threshold (IoU) | `detNms` | 0.70 | 0.70 |
| Duplicate-track threshold (IoU) | `dupIou` | 0.75, same class | 0.75, any class |
| Kalman noise: position, velocity, measurement | `wPos`, `wVel`, `wMeas` | 1/20, 1/160, 0.15 | 1/20, 1/160, 0.15 |
| Class mismatch | `classGate`, `classPenalty` (λ) | match excluded (`classGate = 1`) | penalty 0.5 (`classGate = 0`, `classPenalty = 0.5`) |
| Confirmation | `minHits`, `minHitsHigh`, `confHigh` (τ), `minHitsSec` (T<sub>p</sub>) | 20 frames (`confHigh = 2`, `minHitsSec = -1`: off) | 2 matches if p ≥ 0.8, otherwise 0.6667 s |
| Track lifetime without a match | `maxAge`, `maxAgeSec` (T<sub>a</sub>) | 100 frames (`maxAgeSec = -1`: off) | 3.3333 s |

`track.h` also contains options that were tried during development and are not used in the article (`selfCal`, `probRule`, `reidAfterN`, `minConf`); they are switched off.

## Reproducing the experiments

Each script in `experiments/` describes at the top what it does, what it needs and how it is run. The scripts expect the DeepStream container with the datasets, engines and configuration files under `/workspace`, or in the folder given in the environment variable `P<NN>_W` of each script (for example `P55_W`). Scripts `p44`, `p52`–`p56`, `p69` and `p70` can be rerun in evaluation mode only, without a GPU, with `--eval`.

`pipeline/probe_moj*.py` run the pipeline with our library and `pipeline/probe_to_json*.py` with an NVIDIA tracker; the `_p55` versions set the stream multiplexer to 1280×720, the others to 1920×1080. They are called as `python3 probe_moj_p55.py <video> <tracker config> <output.json> [<nvinfer config>]`. `pipeline/id_smooth.py` provides the loaders used in the evaluation; its ID-smoothing function is not applied in any experiment of the article.

Comments, descriptions and printed output are in English. File names, command-line options and the keys in the JSON result files keep their original Croatian names, so that they match the files written in the runs reported in the article. The options are:

| Option | Meaning | Scripts |
|---|---|---|
| `--eval`, `--analiza` | evaluation only, from existing logs (no GPU) | `p44`, `p52`–`p56`, `p69`, `p70`; `p60_fvessel`, `p64`–`p67`, `p73` |
| `--vrati` | restore `track.h` and the engines after an interrupted run | `p44`, `p52`–`p55`, `p60_fvessel`, `p67`, `p69`; for `p64`–`p66` and `p73` run `p60_fvessel.py --vrati` |
| `--popravi` | repair after two simultaneous runs | `p55` |
| `--provjera`, `--dijagnoza` | quick check of the BoxMOT trackers; full traceback of the first error | `p56` |
| `--det=coco`, `--det=ssmot` | choice of detector | `p60_fvessel`, `p64`, `p73` |
| `--pomak=K` | fixed frame offset K (sensitivity) | `p60_fvessel`, `p64`, `p73` |
| `--samo-fiksne` | run only the 14 fixed-camera ShipsMOT sequences | `p73` |
| `--samo-detektor` | detector check only | `p63` |
| `--fvessel-pomaci`, `--laece`, `--gmm` | additional analyses of the companion calibration study | `p67` |

## Data and models

No data and no model weights are included. See [data/README.md](data/README.md) for how to obtain SSMOT, SMD-Plus, FVessel and ShipsMOT. The YOLO11m and OSNet weights were trained on the SSMOT detection and Re-ID subsets; they are available from the authors upon reasonable request, with the agreement of the author of SSMOT.


## License

MIT, see [LICENSE](LICENSE). The NVIDIA DeepStream SDK, TensorRT and the NVIDIA tracker configuration files are not part of this repository and are subject to their own licenses.
