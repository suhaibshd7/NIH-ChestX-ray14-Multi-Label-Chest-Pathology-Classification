# NIH ChestX-ray14 — Multi-Label Chest Pathology Classification

Multi-label classification of 14 chest pathologies from frontal chest X-rays using a
fine-tuned ResNet-18, validated with bootstrap confidence intervals, subgroup (gender/age)
breakdowns, a tested acquisition-bias hypothesis, and Grad-CAM interpretability checked
quantitatively against NIH's ground-truth bounding boxes.

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
| 10 | Pleural_Thickening | Thickening of the pleural membrane — often post-inflammatory |
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

**Portable AP vs standard PA films — tested, not supported for the hypothesised classes.**
The original concern: severe pathologies (pneumothorax, effusion, edema) disproportionately
appear on portable AP films taken at the bedside for critically ill patients, so the model
could be using film-acquisition type as a proxy for disease severity rather than the
pathology itself. Test-set AUC stratified by `View Position`:

| Class | AP | PA | Gap (AP−PA) |
|---|---|---|---|
| Pneumothorax | 0.8332 | 0.8441 | −0.0109 |
| Effusion | 0.7858 | 0.8493 | −0.0635 |
| Atelectasis | 0.7415 | 0.7581 | −0.0166 |

If acquisition type were inflating these three classes' AUC, the gap should be clearly
positive (AP scoring higher). It's flat-to-negative for all three — PA is equal or better
in every case. **This hypothesis is not supported by the data.** That doesn't rule out
shortcut learning through some other mechanism, but this specific one isn't it.

One class did show a large gap in the full 14-class breakdown: Cardiomegaly (AP 0.8361 vs
PA 0.9293, a −0.093 gap — the largest of any class).
*[Note for author: this is a natural place to add your own clinical read — e.g. whether
cardiothoracic ratio assessment is only considered reliable on PA views due to
magnification differences between AP and PA acquisition, which would make this a real
clinical effect rather than a model artifact.]*

**Support apparatus:** ICU patients with life-threatening conditions routinely have
visible ECG leads, endotracheal tubes, central venous catheters, and nasogastric tubes
on their films. These appear systematically with the most severe diagnoses. The model
may learn to associate the presence of lines and tubes with specific pathologies rather
than the anatomical changes they represent. (Not directly tested here — `View Position`
was a usable proxy for acquisition context; the presence of support apparatus itself
isn't a field in this dataset.)

**Single scanner protocol:** Consistent imaging protocols across one institution mean
the model has not been tested for generalisability to different scanner manufacturers,
kVp settings, or patient demographics.

### 4. Subgroup performance — gender and age

**Gender.** Test AUC by `Patient Gender`, sorted by largest absolute gap:

| Class | F | M | Gap (M−F) |
|---|---|---|---|
| Hernia | 0.9279 | 0.8762 | −0.0517 |
| Fibrosis | 0.7516 | 0.8013 | +0.0497 |
| Nodule | 0.7435 | 0.6962 | −0.0473 |
| Effusion | 0.7994 | 0.8255 | +0.0261 |
| Pleural_Thickening | 0.7665 | 0.7438 | −0.0227 |
| *(remaining 9 classes)* | | | ≤ 0.022 |

No individual-class gap has a confidence interval attached, and the bootstrap CI widths
on the full test set (Results section below) run 0.014–0.08 depending on class — so gaps
in the 0.02–0.05 range here are plausibly within normal sampling noise rather than a real
disparity. Nothing here should be read as a confirmed fairness finding without a CI
computed on the subgroup estimates specifically, which hasn't been done.

**Age.** Test AUC by age band (`<40` / `40-60` / `60+`):

| Class | <40 | 40-60 | 60+ |
|---|---|---|---|
| Cardiomegaly | 0.8727 | 0.8931 | 0.8522 |
| Pneumonia | 0.7139 | 0.6674 | 0.7254 |
| Pleural_Thickening | 0.7822 | 0.7567 | 0.7094 |
| *(remaining 11 classes)* | | | within ~0.03 across bands |

**Hernia's `<40` value (0.2185) is not a fairness finding.** Hernia has only 86
test-set positives total across all ages, and whatever fraction lands in the
youngest band is almost certainly a handful of cases at most, where AUC is
dominated by one or two hard examples rather than measuring anything stable.
It should not be read as a genuine 19x performance gap between age groups.

### 5. Label co-occurrence encodes clinical context

Conditions frequently appear together in specific clinical contexts — effusion with
consolidation, emphysema with fibrosis, edema with cardiomegaly. The model learns
these co-occurrence statistics, which reflect real pathophysiology but also reflect the
NIH patient population specifically. This cannot be separated in analysis.

### 6. Severe class imbalance

| Class | Global Count | Positive rate | pos_weight (Train Set) |
|---|---|---|---|
| Infiltration | 19,870 | 17.7% | 5.2 |
| Effusion | 13,307 | 11.9% | 8.9 |
| Atelectasis | 11,535 | 10.3% | 9.5 |
| Nodule | 6,323 | 5.6% | 17.4 |
| Mass | 5,746 | 5.1% | 20.4 |
| Pneumothorax | 5,298 | 4.7% | 31.6 |
| Consolidation | 4,667 | 4.2% | 29.2 |
| Pleural_Thickening | 3,385 | 3.0% | 38.1 |
| Cardiomegaly | 2,772 | 2.5% | 49.4 |
| Emphysema | 2,516 | 2.2% | 58.2 |
| Edema | 2,303 | 2.0% | 61.2 |
| Fibrosis | 1,686 | 1.5% | 69.2 |
| Pneumonia | 1,353 | 1.2% | 98.6 |
| Hernia | 227 | 0.2% | 638.3 |

Per-class `pos_weight` = neg_count / pos_count calculated explicitly within the training
set partition. Hernia's weight of 638.3 means a missed Hernia contributes 638× more to
the loss than a missed negative.

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
- **Training:** Up to 20 epochs configured, early stopping (patience=4) on val mean AUC.
  Best checkpoint at **epoch 7** (val mean AUC 0.8138); training stopped at epoch 11.
- **Reproducibility:** `torch.manual_seed` + deterministic cuDNN set, in addition to the
  `np.random.seed` used for the train/val split
- **Metric:** AUC-ROC per class and mean AUC — correct for severely imbalanced multi-label tasks

A ResNet-50 run (`nn.Linear(2048, 14)`) is a natural next step but has not been trained —
everything below reflects the ResNet-18 result that was actually run.

### Training dynamics — not fully resolved

Train loss fell from 0.811 (epoch 7, the checkpoint kept) to 0.559 (epoch 11) while val
mean AUC did not improve over that span: 0.8138 → 0.7919 → 0.8128 → 0.8055 → 0.8046, net
negative. That's a genuine, if mild, overfitting signature — the model kept fitting the
training set after it stopped finding anything generalizable in validation. Early stopping
mechanically kept the epoch-7 checkpoint, but nothing here actively addressed the pattern:
no weight decay, no dropout, no additional augmentation was tried to see whether the gap
could be narrowed.

There's a second problem underneath the first. Epochs 5, 6, 7, and 9 scored 0.8117,
0.8135, 0.8138, 0.8128 — a spread of 0.0021 across four epochs. The improvement that
caused epoch 7 to be saved over epoch 6 was 0.0003. Epoch 8 dipped to 0.7919 — about
0.02 below its neighbours — and fully recovered by epoch 9. That pattern (single-epoch
swings larger than the margin deciding which checkpoint gets kept) means epoch 7 is not
demonstrably a stable optimum; it's the highest draw from a plateau of roughly
equivalent checkpoints, evaluated once each on a single fixed validation set with no
uncertainty attached. The same reasoning that motivated bootstrap CIs on the test AUC
below applies here and hasn't been applied.

What this would take to resolve: model selection based on a smoothed multi-epoch average
or a minimum-improvement threshold rather than a single-epoch maximum; weight decay
and/or dropout to test whether the train/val gap narrows; and, ideally, repeating training
across a few different train/val splits to see whether "epoch 7, ~0.814 mean val AUC"
holds up or was specific to this one split. None of that has been done. The result below
should be read as a reasonable ResNet-18 baseline, not a tuned optimum.

---

## Results

Evaluated on the official patient-disjoint test set (25,596 images, 2,797 patients).
95% CIs computed via 1000-sample bootstrap resampling of the test set per class.

| Class | AUC | 95% CI | Test positives |
|---|---|---|---|
| Hernia | 0.9029 | [0.860, 0.941] | 86 |
| Emphysema | 0.8988 | [0.890, 0.909] | 1,093 |
| Cardiomegaly | 0.8781 | [0.868, 0.887] | 1,065 |
| Pneumothorax | 0.8452 | [0.838, 0.853] | 2,661 |
| Edema | 0.8303 | [0.820, 0.842] | 925 |
| Effusion | 0.8150 | [0.809, 0.822] | 4,648 |
| Mass | 0.7814 | [0.769, 0.793] | 1,712 |
| Fibrosis | 0.7798 | [0.759, 0.803] | 435 |
| Pleural_Thickening | 0.7536 | [0.740, 0.767] | 1,143 |
| Atelectasis | 0.7486 | [0.740, 0.758] | 3,255 |
| Consolidation | 0.7246 | [0.714, 0.735] | 1,815 |
| Nodule | 0.7163 | [0.703, 0.730] | 1,615 |
| Pneumonia | 0.6982 | [0.675, 0.721] | 477 |
| Infiltration | 0.6883 | [0.681, 0.695] | 6,088 |
| **Mean AUC** | **0.7901** | | |

Note the CI width scales with test-set positive count, not just the point estimate —
Hernia's is nearly 3x wider than Emphysema's despite a similar AUC, because it has ~13x
fewer positives to estimate from.

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

**Interpretation:** this project's result clears the original 2017 baseline —
which the field has treated as an easy bar to clear for years — and sits in a reasonable
range for a standard architecture without location-awareness, self-supervised
pretraining, or other add-ons. It is roughly 3–4 points of mean AUC below the best
published DenseNet-121/ViT results on this same split. That gap is expected given this
project doesn't use any of the techniques those papers add — it isn't a surprising or
notable result in either direction, and shouldn't be presented as one.

---

## Grad-CAM Interpretability

A qualitative pass (single highest-confidence true positive per class) was run first,
then checked quantitatively against NIH's ground-truth bounding boxes using the
**pointing game** metric (Selvaraju et al.'s own Grad-CAM evaluation): does the heatmap's
single highest-activation pixel fall inside the true lesion box?

**NIH only provides ground-truth boxes for 8 of the 14 classes.** The other six
(Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia) cannot be checked
this way with this dataset — any claim about their localization quality is a qualitative
impression only, and is kept separate from the measured results below rather than
presented at the same confidence level.

### Measured (8 classes with ground-truth boxes)

| Class | n boxes | Pointing-game accuracy |
|---|---|---|
| Cardiomegaly | 146 | 0.925 |
| Infiltration | 123 | 0.382 |
| Pneumonia | 120 | 0.333 |
| Mass | 85 | 0.318 |
| Effusion | 153 | 0.222 |
| Atelectasis | 180 | 0.128 |
| Pneumothorax | 98 | 0.122 |
| Nodule | 79 | 0.051 |

Reading this: Cardiomegaly is genuinely strong. Everything else the model does at best
moderately (Infiltration, Pneumonia, Mass — roughly one in three) and at worst close to
chance for a small target region (Nodule).

Two separate reasons not to take that ranking at face value yet.

First, specific to Cardiomegaly: Grad-CAM heatmaps have a documented tendency toward
central activation regardless of the true lesion, and the heart occupies the middle of
essentially every chest film. Some of the 92.5% may reflect this coincidental spatial
alignment rather than the model precisely reasoning about cardiac silhouette. It's the
strongest result here either way, but "strongest" and "fully explained" aren't the
same claim.

**A more general version of that caveat applies across all 8 classes, not just Cardiomegaly.**
Pointing game's "chance" hit rate scales with box size relative to image area — a larger
ground-truth box is easier to hit by construction, regardless of whether the model is
reasoning about the pathology at all. Cardiomegaly boxes cover a large share of the
cardiac silhouette; Nodule boxes, by clinical definition ≤3cm, cover very little of a
full chest film. Nodule being both the smallest-boxed class and the worst score, and
Cardiomegaly both the largest-boxed and the best, is at least consistent with box size
doing some of the work here — though the middle of the ranking doesn't track size as
cleanly, so it isn't a clean confound end to end.

**This hasn't been checked.** The exact ground-truth box dimensions for every test-set
instance are already present in the data (`bbox_test`'s box coordinates), which would
give the true chance rate directly — no need to lean on published population-average
lesion sizes. That comparison (observed pointing-game accuracy against each class's
mean box-area-to-image-area ratio, i.e. a "lift over chance" figure) hasn't been run.
Until it is, the ranking above should be read as raw pointing-game accuracy, not as a
validated statement that Cardiomegaly localizes well and Nodule doesn't — box size alone
could explain a meaningful part of that gap.

Infiltration's measured pointing-game accuracy (38.2% across 123 images) is moderate,
not strong. A single qualitative example is not a reliable basis for judging
localisation quality — this is exactly why the measured, multi-image result is the
one to trust.

### Unmeasured (6 classes, no ground-truth boxes available)

Qualitative impression only, not verified: Consolidation and Pleural_Thickening appeared,
on the single example inspected, to activate anatomically plausible regions; Emphysema,
Fibrosis, Edema, and Hernia were not closely inspected qualitatively. None of this should
be read with the same confidence as the measured table above.

This general pattern — decent AUC despite frequently poor localisation — is a documented
property of ChestX-ray14, discussed by Oakden-Rayner (2017):
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
tabulate==0.9.0
```

## How to Run

1. Add the [NIH ChestX-ray14 dataset](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized) to your Kaggle notebook
2. Set `MINI_RUN = True` for a fast pipeline check, `False` for the full run reported above
3. Run all cells top to bottom — the notebook includes training, evaluation, Grad-CAM,
   bootstrap CIs, subgroup breakdowns, the view-position hypothesis check, and the
   quantitative Grad-CAM localisation, in one script
4. The final section prints paste-ready markdown tables matching everything in this README
