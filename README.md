# NIH ChestX-ray14 — Multi-Label Chest Pathology Classification

Multi-label classification of 14 chest pathologies from frontal chest X-rays using a
fine-tuned ResNet-50, with Grad-CAM interpretability and honest documentation of
dataset limitations.

---

## The Task

Each chest X-ray is labelled with zero, one, or multiple pathologies simultaneously.
The model outputs 14 independent probabilities — one per condition — rather than a
single class. This is fundamentally different from standard image classification.

---

## The 14 Conditions — Clinical Context

| # | Condition | What it means clinically |
|---|---|---|
| 1 | Atelectasis | Partial or complete lung collapse — airways blocked or compressed |
| 2 | Consolidation | Air spaces filled with fluid or cells — pneumonia-like appearance |
| 3 | Infiltration | Diffuse haziness from fluid or inflammatory cells in the airways |
| 4 | Pneumothorax | Air in the pleural space — lung collapses inward, life-threatening if tension |
| 5 | Edema | Fluid leaking into lung tissue — often from heart failure |
| 6 | Emphysema | Irreversible destruction of air sacs — hyperinflated lungs on X-ray |
| 7 | Fibrosis | Scarring of lung tissue — reduced compliance, honeycombing pattern |
| 8 | Effusion | Fluid accumulation in the pleural space around the lung |
| 9 | Pneumonia | Lung infection — lobar or patchy consolidation |
| 10 | Pleural Thickening | Thickening of the pleural membrane — often post-inflammatory |
| 11 | Cardiomegaly | Enlarged cardiac silhouette — cardiothoracic ratio >0.5 |
| 12 | Nodule | Small rounded opacity ≤3cm — may be benign or malignant |
| 13 | Mass | Larger rounded opacity >3cm — higher malignancy concern |
| 14 | Hernia | Herniation of abdominal contents through the diaphragm |

---

## Dataset

**NIH ChestX-ray14**
112,120 frontal-view chest X-ray images from 30,805 unique patients at the NIH Clinical Center.

> Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale
> Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and
> Localization of Common Thorax Diseases. IEEE CVPR 2017.
> https://arxiv.org/abs/1705.02315

Kaggle (pre-resized 224×224): https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized

---

## Dataset Limitations — Read Before Interpreting Results

These are not minor caveats. They fundamentally affect what this model can and cannot do.

### 1. Labels are NLP-generated, not radiologist-confirmed

Labels were extracted automatically from radiology report text using Natural Language
Processing — not by radiologists manually reviewing images. Estimated accuracy is >90%,
meaning up to 10% of labels may be incorrect. This is called **weakly supervised learning**.
The model is trained on imperfect ground truth and results must be interpreted accordingly.

The original radiology reports are not publicly available, so label errors cannot be
audited or corrected.

### 2. "No Finding" does not mean healthy

"No Finding" means the NLP found none of the 14 target conditions mentioned in the
report — not that the patient is free of disease. The patient may have other pathologies
outside the 14-class vocabulary, or findings the radiologist noted but the NLP missed.
In this dataset, 53.9% of images are labelled "No Finding".

### 3. Single hospital — systematic confounding bias

All images come from one institution: the NIH Clinical Center in Bethesda, Maryland.
This creates consistent, systematic confounders that are not present in real-world deployment:

**Portable AP vs standard PA films:** Severe pathologies (pneumothorax, effusion, edema)
disproportionately appear on portable anteroposterior (AP) films taken at the bedside for
critically ill patients. These films have a characteristic appearance — patient positioning,
mediastinal widening, shorter focus-to-film distance. A model trained on this dataset can
use the film acquisition type as a proxy for disease severity rather than learning the
pathology itself.

**Support apparatus:** ICU patients with life-threatening conditions routinely have
visible ECG leads, endotracheal tubes, central venous catheters, and nasogastric tubes
on their films. These appear systematically with the most severe diagnoses. The model
may learn to associate the presence of lines and tubes with specific pathologies rather
than the anatomical changes they represent.

**Single scanner protocol:** Consistent imaging protocols across one institution mean
the model has not been tested for generalisability to different scanner manufacturers,
kVp settings, or patient demographics.

### 4. Label co-occurrence encodes clinical context

Conditions frequently appear together in specific clinical contexts — effusion with
consolidation, emphysema with fibrosis, edema with cardiomegaly. The model learns
these co-occurrence statistics, which reflect real pathophysiology but also reflect the
NIH patient population specifically. This cannot be separated in analysis.

### 5. Severe class imbalance

| Class | Count | Positive rate | pos_weight |
|---|---|---|---|
| Infiltration | 19,870 | 17.7% | 5.2 |
| Effusion | 13,307 | 11.9% | 8.9 |
| Atelectasis | 11,535 | 10.3% | 9.5 |
| Nodule | 6,323 | 5.6% | 17.4 |
| Mass | 5,746 | 5.1% | 20.4 |
| Pneumothorax | 5,298 | 4.7% | 31.6 |
| Consolidation | 4,667 | 4.2% | 29.2 |
| Pleural Thickening | 3,385 | 3.0% | 38.1 |
| Cardiomegaly | 2,772 | 2.5% | 49.4 |
| Emphysema | 2,516 | 2.2% | 58.2 |
| Edema | 2,303 | 2.0% | 61.2 |
| Fibrosis | 1,686 | 1.5% | 69.2 |
| Pneumonia | 1,353 | 1.2% | 98.6 |
| Hernia | 227 | 0.2% | 638.3 |

Per-class `pos_weight` = neg_count / pos_count per class in the training set.
Hernia's weight of 638.3 means a missed Hernia contributes 638× more to the loss
than a missed negative. Rare class performance should still be treated with caution
given only 122 training positives.

---

## Method

- **Model:** ResNet-50 pretrained on ImageNet, final layer replaced with `nn.Linear(2048, 14)`
- **Output:** 14 independent sigmoid activations — no softmax (classes are not mutually exclusive)
- **Loss:** `BCEWithLogitsLoss` with per-class `pos_weight` (neg_count / pos_count per class)
- **Optimiser:** Adam, lr=1e-4
- **Scheduler:** ReduceLROnPlateau on mean val AUC, patience=2, factor=0.5
- **Augmentation:** RandomHorizontalFlip only — vertical flip is clinically invalid for chest X-rays
- **Split:** Official patient-disjoint train/val/test lists provided with the dataset — Train: 77,994 images (25,208 patients) | Val: 8,530 (2,800 patients) | Test: 25,596 (2,797 patients)
- **Metric:** AUC-ROC per class and mean AUC — correct for severely imbalanced multi-label tasks

---

## Results

Evaluated on the official patient-disjoint test set (25,596 images, 2,797 patients).

Evaluated on the official patient-disjoint test set. Model: ResNet-18, 11 epochs, early stopping on val AUC.

| Class | AUC | NIH Paper (DenseNet-121) |
|---|---|---|
| Atelectasis | 0.7350 | 0.7003 |
| Consolidation | 0.7236 | 0.7032 |
| Infiltration | 0.6865 | 0.6614 |
| Pneumothorax | 0.8512 | 0.7993 |
| Edema | 0.8228 | 0.8052 |
| Emphysema | 0.8948 | 0.8330 |
| Fibrosis | 0.7898 | 0.7859 |
| Effusion | 0.8120 | 0.7585 |
| Pneumonia | 0.6894 | 0.6576 |
| Pleural Thickening | 0.7471 | 0.6835 |
| Cardiomegaly | 0.8590 | 0.8100 |
| Nodule | 0.7006 | 0.6687 |
| Mass | 0.7795 | 0.6933 |
| Hernia | 0.8884 | 0.9165 |
| **Mean AUC** | **0.7843** | **0.7451** |

ResNet-18 baseline exceeds the NIH paper's DenseNet-121 results on 13 of 14 classes.
Hernia is the exception — DenseNet-121 scores higher (0.9165 vs 0.8884), likely due
to the larger model capacity handling the severe class imbalance (638:1) more effectively.

**Interpreting these numbers honestly:**
The NIH paper used a different train/val/test split procedure and evaluated on a subset
of images with additional radiologist annotations. Direct numerical comparison is
approximate. What matters is that the results are in the same range and that the
per-class pattern (Infiltration and Pneumonia hardest, Emphysema and Cardiomegaly
easiest) is consistent with published literature — confirming the pipeline is correct.

---

## Grad-CAM Interpretability

Grad-CAM heatmaps are generated for the highest-confidence true positive per class.
Results fall into three groups:

**Valid localisation:** Cardiomegaly (cardiac silhouette), Pleural Thickening (apical
pleural margin), Infiltration and Consolidation (mid-lower lung zones). The model
attends to the correct anatomical regions.

**Shortcut learning:** Pneumothorax, Effusion, Atelectasis, Emphysema. The model
focuses on the central mediastinum rather than the peripheral pleural space or lung
margins. These classes disproportionately appear on portable ICU films — the model
has learned film type and mediastinal shift as proxies for the pathology.

**Poor localisation:** Nodule, Mass, Pneumonia. Broad diffuse activation rather than
the focal, circumscribed heatmaps these conditions require. Insufficient training
examples for rare, subtle findings.

This pattern — high AUC despite incorrect localisation — is a known property of
ChestX-ray14 and was documented by Oakden-Rayner (2017):
https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/

---

## Requirements

Python 3.10

```
torch==2.0.0
torchvision==0.15.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.2.0
matplotlib==3.7.0
seaborn==0.12.0
Pillow==9.5.0
grad-cam==1.4.8
```

## How to Run

1. Add the [NIH ChestX-ray14 dataset](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized) to your Kaggle notebook
2. Set `MINI_RUN = True` in Cell 2 for a 2-minute pipeline test
3. Set `MINI_RUN = False` and swap `resnet18` → `resnet50` in Cell 13 for the full run
4. Run all cells top to bottom
