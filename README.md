# B-ALL Blood Smear Classification using Deep Learning

A deep learning research project for classifying microscopic blood smear images associated with **B-cell Acute Lymphoblastic Leukemia (B-ALL)** using Convolutional Neural Networks and transfer learning with EfficientNetB0.

The project extends beyond classification accuracy by evaluating:

- duplicate-aware data cleaning
- class-wise performance
- confidence calibration
- confidence-based referral of uncertain cases
- Grad-CAM explainability

> **Research-use disclaimer:** This project is a proof-of-concept for blood-smear image classification and is not intended for clinical diagnosis.

---

## Project Overview

The project was developed in two phases.

### Phase 1 — Baseline CNN

A basic Convolutional Neural Network was implemented as an initial baseline for blood-cell image classification.

### Phase 2 — Transfer Learning with EfficientNetB0

The second phase uses ImageNet-pretrained **EfficientNetB0** with transfer learning and fine-tuning.

The updated research pipeline adds:

- exact duplicate detection using SHA-256 hashes
- duplicate removal before data splitting
- stratified train / validation / test splitting
- train-only class-weight calculation
- frozen test-set evaluation
- per-class precision, recall, and F1-score
- confusion-matrix analysis
- confidence calibration
- confidence-based referral analysis
- Grad-CAM visualization

The current research evaluation uses four dataset classes:

- **Benign**
- **Malignant Early Pre-B**
- **Malignant Pre-B**
- **Malignant Pro-B**

These labels are treated as **dataset classification categories** and should not be interpreted as an independent clinical diagnosis or direct determination of disease stage.

---

## Dataset

This project uses the **Blood Cell Cancer [ALL] - 4 Class** dataset available on Kaggle:

https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class

The dataset contains microscopy images organized into four classes.

### Dataset after duplicate removal

| Class | Unique images |
|---|---:|
| Benign | 512 |
| Early Pre-B | 979 |
| Pre-B | 955 |
| Pro-B | 756 |
| **Total** | **3,202** |

During the corrected evaluation pipeline, **40 exact duplicate images were identified and removed before splitting**.

### Data split

The remaining 3,202 unique images were divided using a stratified image-level split:

| Split | Images |
|---|---:|
| Training | 2,241 |
| Validation | 480 |
| Test | 481 |

The split manifest is saved in:

```text
results/split_manifest.csv
```

The duplicate log is saved in:

```text
results/duplicates_skipped.csv
```

### Important limitation

The dataset copy used in this project does not provide verified patient identifiers suitable for a reliable patient-level split.

Therefore, the current evaluation is **image-level rather than patient-level**.

The reported results should be interpreted as proof-of-concept image-classification performance and **not** as evidence of patient-level clinical generalization.

---

## Model

The primary research model uses **EfficientNetB0** with ImageNet-pretrained weights.

### Model workflow

```text
Blood smear image
        ↓
Preprocessing
        ↓
EfficientNetB0
        ↓
Transfer learning / fine-tuning
        ↓
4-class probability output
        ↓
Benign / Early Pre-B / Pre-B / Pro-B
```

### Training strategy

Training was performed in two stages:

```text
Stage 1
Frozen EfficientNetB0 feature extractor

Stage 2
Partial fine-tuning of EfficientNetB0
```

Class weights were calculated using the **training set only**.

---

## Research Evaluation Pipeline

```text
Exact duplicate detection
        ↓
Duplicate removal
        ↓
Stratified train / validation / test split
        ↓
Transfer learning
        ↓
Fine-tuning
        ↓
Frozen test-set evaluation
        ↓
Per-class metrics
        ↓
Confusion matrix
        ↓
Calibration analysis
        ↓
Confidence-based referral
        ↓
Grad-CAM explainability
```

---

## Results

Evaluation was performed on a frozen test set of **481 images**.

### Overall performance

| Metric | Result |
|---|---:|
| Test Accuracy | **95.43%** |
| Macro F1 | **95.06%** |
| Test Loss | **0.1365** |
| Expected Calibration Error (ECE) | **2.50%** |

### Per-class performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Benign | 0.8875 | 0.9221 | 0.9045 | 77 |
| Early Pre-B | 0.9272 | 0.9524 | 0.9396 | 147 |
| Pre-B | 0.9927 | 0.9510 | 0.9714 | 143 |
| Pro-B | 0.9912 | 0.9825 | 0.9868 | 114 |

The most common error pattern occurred between **Benign and Early Pre-B** images.

The full classification report is available in:

```text
results/classification_report.txt
results/classification_report.json
```

---

## Confusion Matrix

The corrected confusion matrix for the 481-image test set is shown below.

![Confusion Matrix](results/figures/confusion_matrix.png)

The largest confusion pattern was:

```text
Benign ↔ Early Pre-B
```

Pro-B showed the strongest class-wise performance in this test set.

---

## Confidence-Based Referral

The project evaluates whether prediction confidence can be used to identify cases that may benefit from additional human review.

Instead of forcing the model to classify every case automatically, the least-confident predictions can be deferred.

| Referral rate | Referred cases | Retained cases | Accuracy on retained cases |
|---|---:|---:|---:|
| 0% | 0 | 481 | 95.43% |
| 5% | 24 | 457 | 97.16% |
| 10% | 48 | 433 | 98.61% |
| 20% | 96 | 385 | 99.74% |

At a **10% referral rate**, retained-case accuracy increased from **95.43% to 98.61%**.

At a **20% referral rate**, retained-case accuracy reached **99.74%**.

These results describe performance on the **retained cases only** and should not be interpreted as overall clinical-system accuracy.

![Confidence-Based Referral](results/figures/referral_accuracy_curve.png)

---

## Calibration

Model confidence was evaluated using a 10-bin reliability analysis.

The resulting Expected Calibration Error was:

```text
ECE = 2.50%
```

This suggests that model confidence was reasonably aligned with observed accuracy overall.

Some intermediate-confidence bins contained relatively few samples, so those regions should be interpreted cautiously.

![Reliability Diagram](results/figures/reliability_diagram.png)

---

## Grad-CAM Explainability

Grad-CAM was used to visualize image regions contributing to model predictions.

Representative examples were generated for all four dataset classes:

- Benign
- Early Pre-B
- Pre-B
- Pro-B

The visualizations indicate that model attention is generally concentrated in **cell-rich regions**, although some attention maps are diffuse across multiple neighboring cells.

Grad-CAM should be interpreted as a qualitative explanation method and does **not** establish biological causality or prove that highlighted regions correspond to diagnostically definitive cellular features.

![Representative Grad-CAM Examples](results/figures/gradcam_panel_4class.png)

---

## Training Curves

### Accuracy

![Training Accuracy](results/figures/training_accuracy.png)

### Loss

![Training Loss](results/figures/training_loss.png)

---

## Project Structure

```text
aml-detection/
├── models/
│   ├── phase1/
│   │   ├── basic_cnn.h5
│   │   └── best_model.h5
│   └── phase2/
│       ├── best_efficientnetb0.h5
│       └── efficientnetb0_final.h5
│
├── results/
│   ├── classification_report.json
│   ├── classification_report.txt
│   ├── duplicates_skipped.csv
│   ├── split_manifest.csv
│   ├── test_predictions.csv
│   ├── test_summary.json
│   │
│   └── figures/
│       ├── confusion_matrix.png
│       ├── gradcam_panel_4class.png
│       ├── referral_accuracy_curve.png
│       ├── reliability_diagram.png
│       ├── training_accuracy.png
│       └── training_loss.png
│
├── scripts/
│   ├── phase1_basic_model.py
│   ├── phase2_transfer_learning.py
│   └── phase2_transfer_learning_clean.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Reproducibility

The corrected research pipeline is implemented in:

```text
scripts/phase2_transfer_learning_clean.py
```

The pipeline records:

- exact duplicate images removed
- train / validation / test membership
- class weights
- classification metrics
- per-image test predictions
- class probabilities
- prediction confidence

The corresponding outputs are stored in the `results/` directory.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Parisaroozgarian/aml-detection.git
cd aml-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class

The research script expects the following class folders:

```text
Benign
[Malignant] early Pre-B
[Malignant] Pre-B
[Malignant] Pro-B
```

Update the dataset path in the script if necessary.

---

## Running the Research Pipeline

Run:

```bash
python scripts/phase2_transfer_learning_clean.py
```

GPU execution is recommended.

The final research experiments reported in this repository were run using **Google Colab with GPU acceleration**.

---

## Healthcare Research Context

The goal of this work is **not autonomous clinical diagnosis**.

Instead, the project investigates whether deep learning can support research on microscopic blood-image classification by combining:

```text
classification
+
model confidence
+
uncertainty-aware referral
+
visual explainability
```

A possible future human-in-the-loop workflow could be:

```text
Blood smear image
        ↓
AI classification
        ↓
Prediction confidence
        ↓
Higher confidence → classification support
Lower confidence  → additional expert review
```

Any clinical application would require substantially more validation, including patient-level evaluation, external datasets, multicenter data, expert review, and prospective clinical studies.

---

## Limitations

This study has several important limitations:

1. The evaluation is image-level rather than patient-level because verified patient identifiers were not available in the dataset copy used for this study.
2. The data come from a limited source and have not been externally validated across multiple hospitals, microscopes, or imaging systems.
3. The model has not been prospectively evaluated in a clinical workflow.
4. Grad-CAM provides qualitative visual explanations but does not establish biological causality.
5. Confidence-based referral results describe accuracy on retained cases and do not represent complete clinical-workflow performance.
6. The system has not been independently evaluated by hematologists or pathologists.

---

## Future Work

Future work will focus on:

- patient-level evaluation
- external validation
- multicenter datasets
- robustness to staining and acquisition variation
- additional uncertainty-estimation methods
- expert review of Grad-CAM explanations
- prospective evaluation of human-AI collaboration

---

## License

This project is licensed under the MIT License.

---

## Dataset Acknowledgment

Dataset used in this project:

**Blood Cells Cancer (ALL) dataset**

Kaggle:

https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class