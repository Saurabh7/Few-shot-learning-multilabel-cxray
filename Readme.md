## Few Shot Learning Geometric Ensemble for Multi-label Chest X-Rays

The dataset labels are available at: [Google Drive Link](https://drive.google.com/drive/folders/14sE39WIgymO059VhWcv4aj0wf8SzrOT9?usp=sharing)

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
