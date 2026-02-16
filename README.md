# DARENet
Abstract—Road extraction from remote sensing images (RSIs) has long been a promising research direction due to its broad application prospects such as digital twins for smart cities and intelligent transportation systems. However, extracting high-precision roads from RSIs remains challenging for the following reasons:(1) Ambiguity caused by background objects with geometric similarities to roads; (2) Topological discontinuities resulting from background occlusions. To address these issues, we propose a Direction-Aware Road Extraction network (DARENet). First, to address the ambiguity caused by insufficient structural features of extracted roads, we introduce a Multiscale Feature Extraction(MSFE) module with different directions in the skip connections; Second, to alleviate the issue of road topological discontinuities, we propose a Multidirectional Global-Local Fusion (MDGLF)module, which captures the relationship between roads and their environment through global and local processing. Finally,evaluations on three public datasets demonstrate that DARENet outperforms existing models. Our source code is available at https://github.com/ZehuaChenLab/DARENet.

## Overview

![image](https://github.com/CV-mumu/G2L2Net/blob/main/images/1.png)

## LID2Conv

![image](https://github.com/CV-mumu/G2L2Net/blob/main/images/2.png)

## G2L2Attention

![image](https://github.com/CV-mumu/G2L2Net/blob/main/images/3.png)

## Results

![image](https://github.com/CV-mumu/G2L2Net/blob/main/images/4.png)

## Comparation

![image](https://github.com/CV-mumu/G2L2Net/blob/main/images/5.png)

## Ablation Study

![image](https://github.com/CV-mumu/G2L2Net/blob/main/images/6.png)
