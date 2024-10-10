## Few Shot Learning Geometric Ensemble for Multi-label Chest X-Rays
#### Authors: Dana Moukheiber⋆, Saurabh Mahindre⋆, Lama Moukheiber, Mira Moukheiber, Song Wang, Chunwei Ma, George Shih, Yifan Peng, and Mingchen Gao

The dataset labels are available in the labels folder. ~at: [Google Drive Link](https://drive.google.com/drive/folders/14sE39WIgymO059VhWcv4aj0wf8SzrOT9?usp=sharing)~

The weights for pretrained base feature extractor (ResNet) model are available here: [Google Drive Link](https://drive.google.com/file/d/1h0NG_VlF7Ha-IUbq5wAWzIL5Cdz8COY6/view?usp=sharing)

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
