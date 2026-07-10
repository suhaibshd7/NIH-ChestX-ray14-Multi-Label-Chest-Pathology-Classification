# NIH ChestX-ray14 — Multi-Label Chest Pathology Classification

Multi-label classification of 14 chest pathologies from frontal chest X-rays using a
fine-tuned ResNet-18, with Grad-CAM interpretability and documented dataset limitations.

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

*Status: this is currently an untested hypothesis in this project, not a verified finding.
`View Position` (AP/PA) is a field in the dataset. Test-set AUC stratified by view
position, for exactly the classes named above, is added in `additional_analysis.py`
(see Reproducibility section) — results will be added here once run.*

**Support apparatus:** ICU patients with life-threatening conditions routinely have
visible ECG leads, endotracheal tubes, central venous catheters, and nasogastric tubes
on their films. These appear systematically with the most severe diagnoses. The model
may learn to associate the presence of lines and tubes with specific pathologies rather
than the anatomical changes they represent.

**Single scanner protocol:** Consistent imaging protocols across one institution mean
the model has not been tested for generalisability to different scanner manufacturers,
kVp settings, or patient demographics.

### 4. No subgroup (fairness) analysis yet

`Patient Age` and `Patient Gender` are present in the dataset and have not yet been used
to check whether performance is uneven across subgroups. This is a known concern in
chest X-ray AI specifically (e.g. Seyyed-Kalantari et al., "CheXclusion: Fairness Gaps
in Deep Chest X-ray Classifiers," 2020) and is a gap in the current version of this
project. Stratified AUC by gender and age band is added in `additional_analysis.py` —
results will be added here once run.

### 5. Label co-occurrence encodes clinical context

Conditions frequently appear together in specific clinical contexts — effusion with
consolidation, emphysema with fibrosis, edema with cardiomegaly. The model learns
these co-occurrence statistics, which reflect real pathophysiology but also reflect the
NIH patient population specifically. This cannot be separated in analysis.

### 6. Severe class imbalance

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
than a missed negative. With only 122 training positives and (as shown below) no
confidence interval yet on its test AUC, rare-class results should be treated as
provisional, not as a settled performance number.

---

## Method

- **Model:** ResNet-18 pretrained on ImageNet, final layer replaced with `nn.Linear(512, 14)`
- **Output:** 14 independent sigmoid activations — no softmax (classes are not mutually exclusive)
- **Loss:** `BCEWithLogitsLoss` with per-class `pos_weight` (neg_count / pos_count per class)
- **Optimiser:** Adam, lr=1e-4
- **Scheduler:** ReduceLROnPlateau on mean val AUC, patience=2, factor=0.5
- **Augmentation:** RandomHorizontalFlip only — vertical flip is clinically invalid for chest X-rays
- **Split:** Official patient-disjoint train/val/test lists provided with the dataset —
  Train: 77,994 images (25,208 patients) | Val: 8,530 (2,800 patients) | Test: 25,596 (2,797 patients)
- **Training:** 12 epochs run, early stopping (patience=4) on val mean AUC; best checkpoint
  at epoch 8 (val mean AUC 0.8136)
- **Metric:** AUC-ROC per class and mean AUC — correct for severely imbalanced multi-label tasks

A ResNet-50 run (`nn.Linear(2048, 14)`) is a natural next step but has not been trained —
everything below reflects the ResNet-18 result that was actually run.

---

## Results

Evaluated on the official patient-disjoint test set (25,596 images, 2,797 patients).
Numbers below are copied directly from the notebook's own printed output — no results
have been hand-transcribed or taken from a different run.

| Class | AUC | Test positives |
|---|---|---|
| Atelectasis | 0.7457 | — |
| Consolidation | 0.7310 | — |
| Infiltration | 0.6862 | — |
| Pneumothorax | 0.8484 | — |
| Edema | 0.8372 | — |
| Emphysema | 0.9018 | — |
| Fibrosis | 0.8049 | — |
| Effusion | 0.8141 | — |
| Pneumonia | 0.6754 | — |
| Pleural Thickening | 0.7576 | — |
| Cardiomegaly | 0.8703 | — |
| Nodule | 0.6958 | — |
| Mass | 0.7795 | — |
| Hernia | 0.9183 | — |
| **Mean AUC** | **0.7904** | |

*(Test-positive counts and 95% bootstrap confidence intervals per class are added by
`additional_analysis.py` — to be filled in once run; the Hernia and Pneumonia rows in
particular should not be read as precise until a CI is attached to them.)*

### Comparison with published official-split results

Numbers below are compiled from Xiao et al. (2022), *"Delving into Masked Autoencoders
for Multi-Label Thorax Disease Classification,"* Table 6 (arXiv:2210.12843), which
itself aggregates results from ~15 published papers on the same official train/test
split used here.

| Study | Year | Architecture | Mean AUC |
|---|---|---|---|
| Wang et al. (original ChestX-ray14 paper) | 2017 | ResNet-50 | 0.745 |
| Yao et al. | 2018 | ResNet + DenseNet | 0.761 |
| Guendel et al. (DNetLoc, location-aware) | 2018 | DenseNet-121 | 0.807 |
| Baltruschat et al. | 2019 | ResNet-50 | 0.806 |
| Liu et al. | 2022 | DenseNet-121 | 0.818 |
| Best CNN result reported as of 2022 | 2022 | DenseNet-121 | 0.826 |
| Xiao et al. (MAE-pretrained) | 2022 | ViT-B/16 | 0.830 |
| **This project** | 2026 | **ResNet-18** | **0.790** |

**Honest read of this table:** this project's result clears the original 2017 baseline —
which the field has treated as an easy bar to clear for years — and sits in a reasonable
range for a standard architecture without location-awareness, self-supervised
pretraining, or other add-ons. It is roughly 3–4 points of mean AUC below the best
published DenseNet-121/ViT results on this same split. That gap is expected given this
project doesn't use any of the techniques (attention modules, location-aware losses,
domain-specific pretraining) those papers add — it isn't a surprising or notable result
in either direction, and shouldn't be presented as one.

---

## Grad-CAM Interpretability

Grad-CAM heatmaps were generated for the single highest-confidence true positive per
class. **This is a qualitative, anecdotal check (n=1 image per class), not a
measurement** — a visual pattern here is a hypothesis about the model's behaviour, not
a validated finding. With that caveat, the images fell into three visually distinct groups:

**Visually plausible localisation:** Cardiomegaly (cardiac silhouette), Pleural Thickening
(apical pleural margin), Infiltration and Consolidation (mid-lower lung zones). The model
appears to attend to anatomically appropriate regions in these single examples.

**Possible shortcut learning:** Pneumothorax, Effusion, Atelectasis. The heatmap for
these examples centres on the mediastinum rather than the peripheral pleural space or
lung margins that define these conditions. These classes disproportionately appear on
portable ICU films (see Limitation 3 above); one plausible explanation is that the model
is partly using film-acquisition cues as a proxy rather than the pathology itself — but
this has not been tested, only observed on single images.

**Poor localisation:** Nodule, Mass, Pneumonia. Broad, diffuse activation rather than the
focal, circumscribed heatmap these conditions should produce, plausibly reflecting the
smaller number of training examples for these classes.

This general pattern — high AUC despite imperfect localisation — is a documented property
of ChestX-ray14, discussed by Oakden-Rayner (2017):
https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/

**Planned upgrade, not yet done:** NIH released ~880 ground-truth bounding boxes across
8 of the 14 classes specifically for evaluating localisation. `additional_analysis.py`
adds code to threshold each Grad-CAM heatmap, extract a bounding box, and compute
IoU / AP@0.25 / AP@0.50 against these ground-truth boxes for every test image that has
one — turning the n=1 impression above into an actual number per class. Results will be
added here once run. (Methodology follows the same box-extraction approach as Xiao et al.
2022, Table 7, who report this exact metric for DenseNet-121 and ViT-S/16 on this dataset,
for reference.)

---

## Requirements

Python 3.10

```
torch==2.0.0
torchvision==0.15.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.2.0
scipy==1.10.0
matplotlib==3.7.0
seaborn==0.12.0
Pillow==9.5.0
grad-cam==1.4.8
```

## How to Run

1. Add the [NIH ChestX-ray14 dataset](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized) to your Kaggle notebook
2. Set `MINI_RUN = True` in Cell 2 for a 2-minute pipeline test
3. Set `MINI_RUN = False` for the full run (ResNet-18, as reported above)
4. Run all cells top to bottom
5. Append the cells from `additional_analysis.py` for bootstrap CIs, subgroup AUC,
   view-position stratification, and quantitative Grad-CAM localisation

**Note on portability:** the notebook currently hardcodes `/kaggle/input/...` paths and
will not run outside Kaggle without editing `BASE` in Cell 2. `additional_analysis.py`
includes a path fix for this.
