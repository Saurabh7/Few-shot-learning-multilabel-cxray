## Few Shot Learning Geometric Ensemble for Multi-label Chest X-Rays
#### Authors: Dana Moukheiber*, Saurabh Mahindre*, Lama Moukheiber, Mira Moukheiber, Song Wang, Chunwei Ma, George Shih, Yifan Peng, and Mingchen Gao
#### (*) Equal contribution

Paper: [Link](https://link.springer.com/chapter/10.1007/978-3-031-17027-0_12)

#### Dataset
The dataset labels are available in the labels folder.

MIMIC CXR dataset and corresponding images are available at: [physionet.org/content/mimic-cxr/2.0.0/](https://physionet.org/content/mimic-cxr/2.0.0/)
. You need to sign-up as a user on physionet and sign the data use agreement.

#### Model Weights
The weights for pretrained base feature extractor (ResNet) model, NCA model and feature statistics used in Distribution Calibration are available here: [Google Drive Link](https://drive.google.com/drive/folders/17oyc9BoIREtrNhM_-sp67UPltsRBE3Mh?usp=share_link)

#### Notebooks
- Deepvoro.ipynb: Jupyter notebook with code for:
  - Loading model weights
  - DC Voronoi LR
  - NCA Loss finetuning
  - BCE Loss finetuning
  - Episodic evaluation
  - Finetuning
  - Deepvoro ensemble
  - Few-shot evaluation
  
- few_shot_cxray: Module with code for:
  - Residual Network backbone
  - Multilabel NCA Loss definition
  - Dataset loaders
  - Utilities


#### Citation

Please cite our work if you find it useful! 

```
@InProceedings{10.1007/978-3-031-17027-0_12,
author="Moukheiber, Dana
and Mahindre, Saurabh
and Moukheiber, Lama
and Moukheiber, Mira
and Wang, Song
and Ma, Chunwei
and Shih, George
and Peng, Yifan
and Gao, Mingchen",
editor="Nguyen, Hien V.
and Huang, Sharon X.
and Xue, Yuan",
title="Few-Shot Learning Geometric Ensemble for Multi-label Classification of Chest X-Rays",
booktitle="Data Augmentation, Labelling, and Imperfections",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="112--122",
abstract="This paper aims to identify uncommon cardiothoracic diseases and patterns on chest X-ray images. Training a machine learning model to classify rare diseases with multi-label indications is challenging without sufficient labeled training samples. Our model leverages the information from common diseases and adapts to perform on less common mentions. We propose to use multi-label few-shot learning (FSL) schemes including neighborhood component analysis loss, generating additional samples using distribution calibration and fine-tuning based on multi-label classification loss. We utilize the fact that the widely adopted nearest neighbor-based FSL schemes like ProtoNet are Voronoi diagrams in feature space. In our method, the Voronoi diagrams in the features space generated from multi-label schemes are combined into our geometric DeepVoro Multi-label ensemble. The improved performance in multi-label few-shot classification using the multi-label ensemble is demonstrated in our experiments (The code is publicly available at https://github.com/Saurabh7/Few-shot-learning-multilabel-cxray).",
isbn="978-3-031-17027-0"
}

```
