# NIH ChestX-ray14 — Multi-Label Chest Pathology Classification

Multi-label classification of 14 chest pathologies from frontal chest X-rays using a
fine-tuned ResNet-18. Validated with bootstrap confidence intervals, a gender/age
subgroup breakdown, a tested acquisition-bias hypothesis, Grad-CAM interpretability
checked quantitatively against NIH's ground-truth bounding boxes, and two executed
ablations demonstrating why the patient-disjoint split and per-class loss weighting
matter — not just asserting that they do.

Full pipeline: [`chestx-ray14.ipynb`](./chestx-ray14.ipynb)

---

## The Task

Each chest X-ray is labelled with zero, one, or multiple pathologies simultaneously.
The model outputs 14 independent probabilities — one per condition — rather than a
single class, which is why it uses 14 sigmoids rather than a softmax.

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

**NIH ChestX-ray14** — 112,120 frontal-view chest X-rays from 30,805 unique patients
at the NIH Clinical Center.

> Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale
> Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and
> Localization of Common Thorax Diseases. IEEE CVPR 2017.
> https://arxiv.org/abs/1705.02315

Used here: [pre-resized 224×224 Kaggle mirror](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized).

**Note on provenance:** this is a third-party re-upload, resized by a pipeline that
isn't documented. It is not the official NIH 1024×1024 release. This affects exact
reproducibility and comparability with papers that resize from native resolution
themselves — see the literature comparison table below.

---

## Dataset Limitations — Read Before Interpreting Results

**Labels are NLP-extracted, not radiologist-confirmed.** They were pulled from
radiology report text automatically, not assigned by a radiologist reviewing the
image. Estimated label accuracy is >90%, meaning up to 1 in 10 labels may be wrong.
The original reports aren't public, so errors can't be audited. This is standard
practice for this dataset (weak supervision), but it caps how much signal any model
trained on it can extract.

**"No Finding" ≠ healthy.** It means the NLP found none of the 14 target conditions
in the report — not that the patient has no pathology at all. 53.9% of images carry
this label.

**Single hospital — systematic confounding risk.** All images are from one
institution (NIH Clinical Center, Bethesda). Two specific concerns, one tested:

*Acquisition type (portable AP vs. standard PA).* The concern: severe pathologies
(pneumothorax, effusion, edema) disproportionately appear on portable AP films taken
bedside for critically ill patients, so the model could be using acquisition type as
a severity proxy rather than the pathology itself. Tested directly — test AUC
stratified by `View Position`:

| Class | AP | PA | Gap (AP−PA) |
|---|---|---|---|
| Pneumothorax | 0.8127 | 0.8399 | −0.0272 |
| Effusion | 0.7832 | 0.8476 | −0.0643 |
| Atelectasis | 0.7415 | 0.7608 | −0.0194 |

If acquisition type were inflating these three classes, the gap should be clearly
positive (AP scoring higher). It's negative for all three — PA does equal or better
in every case. **This specific hypothesis is not supported.** That doesn't rule out
shortcut learning through some other mechanism, but this one isn't it.

One class *did* show a large gap in the full 14-class breakdown: Cardiomegaly (AP
0.8361 vs. PA 0.9293, a −0.093 gap, the largest of any class). **Clinical read:** this
is plausibly a known limitation of AP films rather than a model artifact.
Cardiothoracic ratio — the standard criterion for cardiomegaly — is validated for PA
views taken at a standardized distance. AP views (typically portable bedside films)
place the heart closer to the X-ray source, causing variable magnification of the
cardiac silhouette, so the ground-truth labels themselves are noisier on AP films.
A noisier label signal would lower measured AUC regardless of what the model is doing
internally. Consistent with the observed gap, not proof of it.

*Support apparatus* (ECG leads, tubes, catheters — common on the sickest patients'
films) is a plausible second confound but wasn't directly tested; it isn't a field in
this dataset, and `View Position` was the available proxy for acquisition context.

*Single scanner protocol* means generalizability to other institutions, scanners, or
demographics is untested — see Ablation/Results caveats below.

**Subgroup performance.** Test AUC by gender and age band, largest gaps first:

| Class | F | M | Gap (M−F) |
|---|---|---|---|
| Fibrosis | 0.7589 | 0.8388 | +0.0799 |
| Hernia | 0.9568 | 0.8768 | −0.0800 |
| Nodule | 0.7379 | 0.6994 | −0.0385 |
| Effusion | 0.7943 | 0.8221 | +0.0278 |
| *(remaining 10 classes)* | | | ≤ 0.018 |

No per-subgroup confidence interval is computed. Bootstrap CI widths on the full test
set (see Results) run 0.013–0.08 depending on class, so most gaps here are plausibly
within normal sampling noise rather than a confirmed disparity — descriptive, not a
fairness finding.

Age band (`<40` / `40-60` / `60+`) shows a similar pattern, with one artifact worth
flagging explicitly: Hernia's `<40` AUC (0.5036) is not a real effect — this band has
very few Hernia-positive test cases, and AUC on a handful of examples is dominated by
one or two hard cases rather than measuring anything stable.

**Label co-occurrence** reflects real clinical patterns (effusion with consolidation,
edema with cardiomegaly) but also reflects this specific patient population — not
separable from the model's raw performance.

**Severe class imbalance:**

| Class | Count | Rate | pos_weight |
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

`pos_weight` = neg_count / pos_count, computed on the train split only. Ablation B
below tests what happens if you skip this.

---

## Method

- **Model:** ResNet-18, ImageNet-pretrained, final layer replaced with `Linear(512, 14)`
- **Output:** 14 independent sigmoids — not softmax, since conditions aren't mutually exclusive
- **Loss:** `BCEWithLogitsLoss` with per-class `pos_weight`
- **Augmentation:** horizontal flip only (p=0.5), nothing else. A mirrored chest is
  still anatomically valid and approximates real variation in patient positioning.
  Vertical flip or arbitrary rotation would place the heart, gastric bubble, and
  diaphragm curvature in orientations that never occur in an upright film — the
  network could end up learning the augmentation artifact instead of the pathology.
- **Optimiser:** Adam, lr=1e-4, `ReduceLROnPlateau` on val mean AUC (patience=2, factor=0.5)
- **Split:** official NIH patient-disjoint train/test lists, with train further split
  90/10 by patient for validation. Train: 77,994 images (25,208 pt) | Val: 8,530
  (2,800 pt) | Test: 25,596 (2,797 pt). Verified disjoint with an explicit assert
  (Ablation A below shows what happens without this).
- **Training:** up to 20 epochs, early stopping (patience=4) on val mean AUC. Best
  checkpoint at **epoch 6** (val mean AUC 0.8179); training stopped at epoch 10.
- **Reproducibility:** fixed seeds (`torch.manual_seed`, `np.random.seed`) +
  deterministic cuDNN. **Verified, not just configured** — the full pipeline was rerun
  end-to-end and reproduced every reported number (main results, both ablations,
  per-epoch loss/AUC) to 4 decimal places.

**Training dynamics — not fully resolved.** Val AUC plateaus in the 0.80–0.82 range
from epoch 4 onward, with epoch-to-epoch noise (±0.01–0.02) larger than the margin
that decided which checkpoint got saved. Read "epoch 6" as *best observed on this
run*, not a precisely tuned optimum — no weight decay, dropout, or multi-seed
averaging was used to test whether the plateau narrows.

---

## Results

Test set: 25,596 images, 2,797 patients, patient-disjoint from train/val.
95% CIs via 1000-sample bootstrap resampling per class.

| Class | AUC | 95% CI | Test positives |
|---|---|---|---|
| Hernia | 0.9189 | [0.880, 0.954] | 86 |
| Emphysema | 0.8822 | [0.872, 0.893] | 1,093 |
| Cardiomegaly | 0.8666 | [0.857, 0.877] | 1,065 |
| Pneumothorax | 0.8339 | [0.826, 0.841] | 2,661 |
| Edema | 0.8334 | [0.822, 0.844] | 925 |
| Effusion | 0.8107 | [0.805, 0.817] | 4,648 |
| Fibrosis | 0.8039 | [0.783, 0.825] | 435 |
| Mass | 0.7782 | [0.767, 0.788] | 1,712 |
| Pleural_Thickening | 0.7568 | [0.743, 0.771] | 1,143 |
| Atelectasis | 0.7499 | [0.741, 0.759] | 3,255 |
| Consolidation | 0.7242 | [0.714, 0.734] | 1,815 |
| Nodule | 0.7151 | [0.702, 0.728] | 1,615 |
| Infiltration | 0.6955 | [0.688, 0.703] | 6,088 |
| Pneumonia | 0.6806 | [0.657, 0.706] | 477 |
| **Mean AUC** | **0.7893** | | |

CI width scales with test-set positive count, not just the point estimate — Hernia's
is ~4x wider than Emphysema's at a similar AUC, because it has ~13x fewer positives.

### Comparison with published official-split results

Compiled from Xiao et al. (2022), *"Delving into Masked Autoencoders for Multi-Label
Thorax Disease Classification"* (arXiv:2210.12843), Table 6, which aggregates ~15
published results on the same official train/test split used here.

| Study | Year | Architecture | Mean AUC |
|---|---|---|---|
| Wang et al. (original ChestX-ray14) | 2017 | ResNet-50 | 0.745 |
| Yao et al. | 2018 | ResNet + DenseNet | 0.761 |
| Guendel et al. (DNetLoc, location-aware) | 2018 | DenseNet-121 | 0.807 |
| Baltruschat et al. | 2019 | ResNet-50 | 0.806 |
| Liu et al. | 2022 | DenseNet-121 | 0.818 |
| Best CNN result, as of 2022 | 2022 | DenseNet-121 | 0.826 |
| Xiao et al. (MAE-pretrained) | 2022 | ViT-B/16 | 0.830 |
| **This project** | 2026 | **ResNet-18** | **0.789** |

Clears the 2017 baseline the field has used as an easy bar for years, and sits
roughly 3–4 points below the best published DenseNet-121/ViT results — expected,
since this project doesn't use location-awareness, self-supervised pretraining, or
any of the other add-ons those papers use. Not a surprising result in either
direction.

---

## Ablations

Both run on a reduced subsample (6,000 train / 1,200 val / 1,200 test images) for 5
epochs — enough to isolate one variable cheaply, not enough to be a tuned result.
Both are executed in the notebook, not just described.

### Ablation A — does the patient-disjoint split actually matter?

Same pool of images, split two ways: honestly (by patient, matching the main
pipeline) and naively (by image, ignoring which patient each image belongs to).

**The clean evidence — deterministic, no training noise involved:**

> **454 of 1,200 leaked-split test images (38%) belong to a patient who also has
> images in the leaked-split train set.** The honest split has zero such overlap by
> construction (394 patients affected).

**The consequence — mean test AUC:**

| | Honest split | Leaked split |
|---|---|---|
| Mean AUC (all 14 classes) | 0.7026 | 0.7299 |

Delta: +0.027. That number is muddier than it looks — Hernia's test subsample had
only 2 positive images, far too few for a stable AUC estimate, and it swung **−0.31**
in the *opposite* direction, dragging the naive mean down. Excluding Hernia (and any
class with <15 test positives): **mean delta +0.054** across the remaining 13 classes,
12 of which moved in the direction leakage predicts, several substantially
(Consolidation +0.18, Emphysema +0.12, Pleural_Thickening +0.09).

The patient-overlap count is the number to lead with. It doesn't depend on training
noise, subsample size, or which epoch got saved — it's just counting.

### Ablation B — does per-class pos_weight actually matter?

Same honest split, same 5 epochs, weighted vs. unweighted `BCEWithLogitsLoss`:

| Class | AUC (weighted) | AUC (unweighted) | Sensitivity@0.5 (weighted) | Sensitivity@0.5 (unweighted) |
|---|---|---|---|---|
| Pneumonia | 0.5821 | 0.5263 | 0.1667 | **0.0000** |
| Fibrosis | 0.7644 | 0.6858 | 0.3600 | **0.0000** |
| Cardiomegaly | 0.7566 | 0.7205 | 0.4681 | **0.0000** |

Sensitivity@0.5 (fraction of true positives actually flagged at a normal decision
threshold) hits exactly zero for all three tracked classes without weighting — the
unweighted model never once flagged a true positive for Pneumonia, Fibrosis, or
Cardiomegaly at threshold 0.5, despite AUC still looking passable. AUC is
threshold-independent (rank-based), so it can look reasonable even when the model has
learned to never actually call anything positive.

AUC also dropped meaningfully unweighted (not just the operating threshold) — under a
5-epoch budget, the unweighted loss likely gives the model too little gradient signal
from rare-class examples to learn to rank them well either, not only to threshold them
well.

---

## Grad-CAM Interpretability

Checked quantitatively against NIH's ground-truth bounding boxes using the
**pointing game** metric (Selvaraju et al.'s own Grad-CAM evaluation): does the
heatmap's single highest-activation pixel fall inside the true lesion box?

**NIH only provides ground-truth boxes for 8 of the 14 classes.** The other six can't
be checked this way with this dataset.

| Class | n boxes | Pointing-game accuracy |
|---|---|---|
| Cardiomegaly | 146 | 0.959 |
| Infiltration | 123 | 0.455 |
| Pneumonia | 120 | 0.442 |
| Effusion | 153 | 0.333 |
| Mass | 85 | 0.306 |
| Atelectasis | 180 | 0.250 |
| Pneumothorax | 98 | 0.082 |
| Nodule | 79 | 0.038 |

Cardiomegaly is genuinely strong; everything else the model does at best moderately,
Nodule close to chance for a small target region. Two caveats:

Pointing-game "chance" scales with box size relative to image area — Cardiomegaly
boxes are large by definition, Nodule boxes are small (≤3cm clinically). Some of this
ranking is plausibly box size, not purely localization quality. Not corrected for
here (would need a box-area-normalized "lift over chance" — the box dimensions are
already in the data, this just hasn't been computed).


---

## Reproducibility

Given the fixed seeds and deterministic cuDNN settings, rerunning this notebook
end-to-end reproduces every reported number exactly — this was verified by rerunning
the full pipeline twice and diffing the outputs, not assumed from the seed
configuration alone.

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

Approximate — not a verified pin of the exact Kaggle session that produced these
results. For an exact match, `pip freeze` from that session.

## How to Run

1. Add the [NIH ChestX-ray14 dataset](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized) to a Kaggle notebook (GPU enabled)
2. Run all cells top to bottom. `MINI_RUN = True` in the config cell for a fast
   pipeline check; `False` for the full run reported here
3. `SKIP_MAIN_TRAINING_IF_CHECKPOINT_EXISTS = True` will load a saved `main_best.pth`
   instead of retraining, if one exists in the working directory
4. Ablation cells (13–14) run automatically after the main pipeline — no separate
   trigger needed
5. The final cell prints paste-ready markdown tables matching everything above

---

## License

MIT License

Copyright (c) 2026 . Suhaib Shdefat

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
