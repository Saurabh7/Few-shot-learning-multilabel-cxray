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
