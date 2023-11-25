# Transfer Learning for Computation Pathology

## Introduction
This work-in-progress repository focuses on investigating the application of a ResNet model, which has been pretrained on digital histopathology images, as encoder for a UNet segmentation framework. The training of the encoder utilized SimCLR, a technique for contrastive self-supervised learning, and was done by Ciga, O., Xu, T., & Martel, A. L. (2022). _Self supervised contrastive learning for digital histopathology. Machine Learning with Applications, 7, 100198._ ([paper](https://arxiv.org/pdf/2011.13971.pdf), [GitHub](https://github.com/ozanciga/self-supervised-histopathology)).

## Dataset
The dataset comes from the _CoNIC: Colon Nuclei Identification and Counting Challenge_, a competition that aims to develop algorithms for accurately detecting and counting nuclei in digital pathology images of colon tissue samples. The CONIC dataset consists in 4,981 tiles of size 256×256 pixel. These tiles include ground truth masks with two channels: one for semantic segmentations of various nuclei types (neutrophil, epithelial, lymphocyte, plasma cell, eosinophil, and connective) and another for instance segmentation, assigning each nucleus a unique label ID.

Out of the six cell types in the dataset, a single biomarker is retained to simplify the task into binary segmentation. Plasma cells (label ID = 4) are chosen due to their rarity. Similarly, eosinophils (label ID = 5) could be selected for the same reason. Conversely, the epithelium class (label ID = 2) is characterized by large, abundant objects, making it a straightforward target for benchmarking.

## Model
The model adopts a UNet architecture, wherein the pretrained encoder can be pruned of a variable number of final layers to yield feature maps of differing shapes.

| n layers to prune | feature maps shape |
|-------------------|--------------------|
| 1                 | [512, 1, 1]        |
| 2                 | [512, 8, 8]        |
| 4                 | [256, 16, 16]      |
| 4                 | [128, 32, 32]      |
| 5                 | [64, 64, 64]       |

The decoder dynamically adjusts the number of hidden layers based on the encoder output's shape, ensuring it consistently returns binary masks with a size of 256x256 pixels.

## Preliminary results
Employing a pretrained ResNet encoder accelerates training compared to using an encoder with randomly initialized weights.