# Configuration files

| File | Use |
|---|---|
| `config_infer_11m.txt` | `nvinfer` configuration of the SSMOT-trained YOLO11m (TensorRT FP16, input 640, confidence threshold 0.25, NMS IoU 0.45), exported with [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo) |
| `config_infer_coco11m_640.txt` | The same for the COCO-pretrained YOLO11m restricted to the class boat (SMD-Plus and FVessel) |
| `config_infer_boat.txt` | Configuration of the earlier detector, which `experiments/p44_mdpi_11m.py` copies to create `config_infer_11m.txt` |

Paths inside the files point to `/workspace`; adapt them to your installation.

## NVIDIA tracker configurations

The configuration files of the NVIDIA trackers are distributed with the DeepStream SDK under NVIDIA's proprietary license and are therefore not included. To reproduce the four NVIDIA trackers of the article, start from the reference files of DeepStream 7.0 and set the parameters listed in the table below. Our library itself does not read a configuration file; its parameters are in `tracker/src/track.h`.

- **NvSORT** — NvSORT reference configuration.
- **NvDeepSORT** — NvDeepSORT configuration with the vessel OSNet engine as the Re-ID model.
- **NvDCF (reidType 1)** — NvDCF configuration with the visual tracker switched off (`visualTrackerType: 0`) and the Re-ID model used in association (`reidType: 1`).
- **NvDCF (reidType 2)** — NvDCF configuration with the correlation-filter visual tracker (`visualTrackerType: 1`) and the Re-ID model used for re-association (`reidType: 2`).

All four associate only detections of the same class (`checkClassMatch: 1`); the class-agnostic test of the article sets `checkClassMatch: 0` and changes nothing else (`experiments/p65_klasa.py`, `experiments/p66_klasa_ssmot.py`).

Values of every parameter set in the four files used in the article (`–`: not present in that file, so the default of the library applies):

| Section | Parameter | NvSORT | NvDeepSORT | NvDCF (reidType 1) | NvDCF (reidType 2) |
|---|---|---|---|---|---|
| BaseConfig | `minDetectorConfidence` | 0.1345 | 0.0762 | 0.1894 | 0.1894 |
| TargetManagement | `enableBboxUnClipping` | 0 | – | 1 | 1 |
| TargetManagement | `maxTargetsPerStream` | 300 | 150 | 150 | 150 |
| TargetManagement | `minIouDiff4NewTarget` | 0.5780 | 0.9847 | 0.3686 | 0.3686 |
| TargetManagement | `minTrackerConfidence` | 0.8216 | 0.4314 | 0.1513 | 0.1513 |
| TargetManagement | `probationAge` | 5 | 2 | 2 | 2 |
| TargetManagement | `maxShadowTrackingAge` | 26 | 68 | 42 | 42 |
| TargetManagement | `earlyTerminationAge` | 1 | 1 | 1 | 1 |
| TrajectoryManagement | `useUniqueID` | 0 | 0 | 0 | 0 |
| DataAssociator | `dataAssociatorType` | 0 | 0 | 0 | 0 |
| DataAssociator | `associationMatcherType` | 1 | 1 | 1 | 1 |
| DataAssociator | `checkClassMatch` | 1 | 1 | 1 | 1 |
| DataAssociator | `minMatchingScore4Overall` | 0.2543 | 0.1794 | 0.0222 | 0.0222 |
| DataAssociator | `minMatchingScore4SizeSimilarity` | 0.4019 | 0.3291 | 0.3552 | 0.3552 |
| DataAssociator | `minMatchingScore4Iou` | 0.2159 | 0.2364 | 0.0548 | 0.0548 |
| DataAssociator | `matchingScoreWeight4SizeSimilarity` | 0.1365 | 0.7178 | 0.6003 | 0.6003 |
| DataAssociator | `matchingScoreWeight4Iou` | 0.3836 | 0.4551 | 0.4033 | 0.4033 |
| DataAssociator | `tentativeDetectorConfidence` | 0.2331 | 0.2479 | 0.1024 | 0.1024 |
| DataAssociator | `minMatchingScore4TentativeIou` | 0.2867 | 0.2376 | 0.2852 | 0.2852 |
| DataAssociator | `usePrediction4Assoc` | 1 | – | – | – |
| StateEstimator | `stateEstimatorType` | 2 | 2 | 1 | 1 |
| StateEstimator | `noiseWeightVar4Loc` | 0.0301 | 0.0503 | – | – |
| StateEstimator | `noiseWeightVar4Vel` | 0.0017 | 0.0037 | – | – |
| StateEstimator | `useAspectRatio` | 1 | 1 | – | – |
| TargetManagement | `preserveStreamUpdateOrder` | – | 0 | 0 | 0 |
| DataAssociator | `thresholdMahalanobis` | – | 12.1875 | – | – |
| DataAssociator | `minMatchingScore4ReidSimilarity` | – | 0.7505 | – | – |
| DataAssociator | `matchingScoreWeight4ReidSimilarity` | – | 0.3197 | – | – |
| ReID | `reidType` | – | 1 | 1 | 2 |
| ReID | `batchSize` | – | 10 | 100 | 100 |
| ReID | `workspaceSize` | – | 1000 | 1000 | 1000 |
| ReID | `reidFeatureSize` | – | 512 | 512 | 512 |
| ReID | `reidHistorySize` | – | 100 | 100 | 100 |
| ReID | `inferDims` | – | [3, 256, 128] | [3, 256, 128] | [3, 256, 128] |
| ReID | `networkMode` | – | 1 | 1 | 1 |
| ReID | `inputOrder` | – | 0 | 0 | 0 |
| ReID | `colorFormat` | – | 0 | 0 | 0 |
| ReID | `offsets` | – | [123.6750, 116.2800, 103.5300] | [123.6750, 116.2800, 103.5300] | [123.6750, 116.2800, 103.5300] |
| ReID | `netScaleFactor` | – | 0.01735207 | 0.01735207 | 0.01735207 |
| ReID | `keepAspc` | – | 1 | 1 | 1 |
| ReID | `addFeatureNormalization` | – | 1 | 1 | 1 |
| ReID | `onnxFile` | – | `<vessel OSNet>` | – | – |
| ReID | `modelEngineFile` | – | `<vessel OSNet>` | `<vessel OSNet>` | `<vessel OSNet>` |
| TrajectoryManagement | `enableReAssoc` | – | – | 1 | 1 |
| TrajectoryManagement | `minMatchingScore4Overall` | – | – | 0.6622 | 0.6622 |
| TrajectoryManagement | `minTrackletMatchingScore` | – | – | 0.2940 | 0.2940 |
| TrajectoryManagement | `minMatchingScore4ReidSimilarity` | – | – | 0.0771 | 0.0771 |
| TrajectoryManagement | `matchingScoreWeight4TrackletSimilarity` | – | – | 0.7981 | 0.7981 |
| TrajectoryManagement | `matchingScoreWeight4ReidSimilarity` | – | – | 0.3848 | 0.3848 |
| TrajectoryManagement | `minTrajectoryLength4Projection` | – | – | 34 | 34 |
| TrajectoryManagement | `prepLength4TrajectoryProjection` | – | – | 58 | 58 |
| TrajectoryManagement | `trajectoryProjectionLength` | – | – | 33 | 33 |
| TrajectoryManagement | `maxAngle4TrackletMatching` | – | – | 67 | 67 |
| TrajectoryManagement | `minSpeedSimilarity4TrackletMatching` | – | – | 0.0574 | 0.0574 |
| TrajectoryManagement | `minBboxSizeSimilarity4TrackletMatching` | – | – | 0.1013 | 0.1013 |
| TrajectoryManagement | `maxTrackletMatchingTimeSearchRange` | – | – | 27 | 27 |
| TrajectoryManagement | `trajectoryProjectionProcessNoiseScale` | – | – | 0.0100 | 0.0100 |
| TrajectoryManagement | `trajectoryProjectionMeasurementNoiseScale` | – | – | 100 | 100 |
| TrajectoryManagement | `trackletSpacialSearchRegionScale` | – | – | 0.0100 | 0.0100 |
| TrajectoryManagement | `reidExtractionInterval` | – | – | 8 | 8 |
| DataAssociator | `minMatchingScore4VisualSimilarity` | – | – | 0.5043 | 0.5043 |
| DataAssociator | `matchingScoreWeight4VisualSimilarity` | – | – | 0.3951 | 0.3951 |
| StateEstimator | `processNoiseVar4Loc` | – | – | 6810.8668 | 6810.8668 |
| StateEstimator | `processNoiseVar4Size` | – | – | 1541.8647 | 1541.8647 |
| StateEstimator | `processNoiseVar4Vel` | – | – | 1348.4874 | 1348.4874 |
| StateEstimator | `measurementNoiseVar4Detector` | – | – | 100.0000 | 100.0000 |
| StateEstimator | `measurementNoiseVar4Tracker` | – | – | 293.3238 | 293.3238 |
| VisualTracker | `visualTrackerType` | – | – | 0 | 1 |
| VisualTracker | `useColorNames` | – | – | 1 | 1 |
| VisualTracker | `useHog` | – | – | 1 | 1 |
| VisualTracker | `featureImgSizeLevel` | – | – | 3 | 3 |
| VisualTracker | `featureFocusOffsetFactor_y` | – | – | -0.1054 | -0.1054 |
| VisualTracker | `filterLr` | – | – | 0.0767 | 0.0767 |
| VisualTracker | `filterChannelWeightsLr` | – | – | 0.0339 | 0.0339 |
| VisualTracker | `gaussianSigma` | – | – | 0.5687 | 0.5687 |
| ReID | `minVisibility4GalleryUpdate` | – | – | 0.6 | 0.6 |
