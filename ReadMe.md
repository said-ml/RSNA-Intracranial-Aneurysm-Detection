# RSNA 2025 Intracranial Aneurysm Detection – End-to-End Deep Learning Baseline 

This repository contains my full **end-to-end pipeline** for the **RSNA (Radiological Society of North America) Kaggle competition**, which focus Detect the presence and location of intracranial aneurysms in multimodal imaging data.

Although the final leaderboard score was modest, this project represents a critical milestone in my deep learning journey — the first complete competition workflow I built and optimized independently.

---

## Overview

**Competition goal:**  
Detect the presence and location of intracranial aneurysms in multimodal imaging data using deep learning models trained on large-scale DICOM datasets.
for more inforamation about the competition check out :https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection
**Main challenges:**
- High data variability (image modality, size, contrast)
- Data Noisy and some samples are corrupted or some elements within samples are mismesured
- Multi-class imbalance some alteries has few samples compared to others.
- Generalization gap between public and private datasets that revael real overfitting
- Limited compute for large-scale augmentation and ensembles in the case of using 3D models, we must apply Mixed Precision and reduce batch size to 2 or 1.
- In my case weak segmentation , pseudo segmentations that generated from train_localizers.csv are useless (in fact they are not useless , unfortunetely i am figure  out how to use them effectively)

---

##  My Approach:

|--------|--------------|
| **Data Preprocessing** | such i've limited SSD storage , i convert .dcm files ( >300GB ) TO .npz files and compress them (<19GB)  using kaggle notebook,
here the data compred and script python use it for that https://www.kaggle.com/code/saidkoussi/data-preaparation.
| **Model Architecture** | CNN backbone (`resnet50`, `efficientnet_b3`), pretrained on ImageNet, fine-tuned with mixed precision (AMP). |
| **Training Framework** | Custom PyTorch Trainer with: EMA, gradient accumulation
| **Loss Functions** | Weighted BCE + Focal loss combination for class imbalance handling. |
| **Optimization** | AdamW optimizer, cosine scheduler, batch size 2 for 3D (16, 256, 256) models, and 16 for 2.5D (7, 500, 500)models, and 32 for 2D(1, 500., 500) models


---

## 📈 Results

| Metric | My Result | Golden Zone (Top Teams) |
|---------|------------|-------------------------|
| CV (Cross-Validation) | **0.60** | **0.90+** |
| LB (Leaderboard) | **0.53** | **0.86+** |

> My solution achieved stable validation but suffered from a large generalization gap — highlighting dataset leakage prevention, model regularization, and augmentation diversity as key areas for improvement.

---

## 🧩 Key Learnings

1. **True generalization beats local CV tuning.**
2. **Medical imaging demands heavy augmentations** and domain-specific normalization.
3. **Validation design defines your ceiling** — wrong splits can destroy leaderboard correlation.
4. **Automation matters**: efficient pipelines and logging accelerate iteration speed.

---

## ⚙️ Tools & Stack

- **Language:** Python 3.10  
- **Frameworks:** PyTorch, Albumentations, scikit-learn  
- **Logging:** Weights & Biases  
- **Hardware:GPU RTX 4070 (12GB VRAM)  
             CPU Ryzen 7 
             RAM = 32gb (2*16GB)
-            SSD : 1TB
- **Environment:** Ubuntu 22.04, CUDA 12.3  

---

##  Next Steps

- Implement **Vision Transformers (ViT / ConvNeXt)** for higher representational power.  
- Integrate **self-supervised pretraining (DINO / MAE)** for medical data.  
- Expand **augmentation pipeline** with MixUp, CutMix, CLAHE, and RandAugment.  
- Introduce **pseudo-labeling** and external data for semi-supervised training.  

---

##  Acknowledgements

Grateful to the **RSNA organizers**, **Kaggle community notebooks**, and **open-source contributors** for the foundational ideas that accelerated this learning journey.

---

##  Author

**Said Koussi**  
Deep Learning Engineer (in training)  
Kaggle Expert – 2× Bronze Medals  
📧 saidkoussi2@gmail.com
🌐 https://www.kaggle.com/saidkoussi

---

