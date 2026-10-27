# Data

No data are included in this repository: no video frames, no ground truth, no exported detections and no model weights.

| Dataset | Use in the article | How to obtain |
|---|---|---|
| SSMOT (Split port, 18 annotated sequences; detection and Re-ID subsets) | Main comparison on 16 sequences, detector and appearance-model training | Described in I. Marin's doctoral thesis (University of Split, 2026, https://repozitorij.fesb.unist.hr/object/fesb:2684); access is governed by its author |
| SMD-Plus (corrected Singapore Maritime Dataset, 40 on-shore sequences) | Evaluation with frozen parameters | Public; see J.-H. Kim et al., J. Mar. Sci. Eng. 10(3), 377, 2022 |
| FVessel (10 river clips, annotated once per second) | Evaluation with frozen parameters | Public; see Y. Guo et al., IEEE Trans. Intell. Transp. Syst. 24(11), 12779–12792, 2023 |

SMD-Plus provides boxes and classes but no track identities. `experiments/p63_smd_gt.py` derives the identities used in the article; the derived files are not distributed here and are available from the authors upon reasonable request, after agreement with the SMD-Plus authors.
