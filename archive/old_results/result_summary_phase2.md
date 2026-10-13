# AML Detection Model Results – Phase 2 (Transfer Learning)

## Model Performance
- **Test Accuracy:** 0.5688 (56.88%)
- **Test Loss:** 1.1069
- **Best Model:** EfficientNetB0

## Classification Report
```
                         precision    recall  f1-score   support

                 Benign       0.43      0.45      0.44        77
[Malignant] early Pre-B       0.67      0.03      0.05       147
      [Malignant] Pre-B       0.70      0.83      0.76       143
      [Malignant] Pro-B       0.52      1.00      0.68       120

               accuracy                           0.57       487
              macro avg       0.58      0.58      0.48       487
           weighted avg       0.60      0.57      0.48       487
```

## Figures
- Confusion Matrix: `results/figures/confusion_matrix_phase2.png`
- Predictions: `results/figures/predictions_phase2.png`
- Training History: `results/figures/transfer_learning_history.png`

## Web Dashboard
- Dashboard: `results/web/phase2_dashboard.html`

---

Phase 2 completed successfully!  
Models saved in: `models/phase2`  
Results saved in: `results/figures`  
Web dashboard saved in: `results/web`

---

## Training Log

```
TensorFlow: 2.13.0
GPU Available: False
Phase 2: Transfer Learning for AML Detection
============================================================

Training with EfficientNetB0...

Loaded 512 images for Benign
Loaded 979 images for [Malignant] early Pre-B
Loaded 955 images for [Malignant] Pre-B
Loaded 796 images for [Malignant] Pro-B
Total images loaded: 3242
Image shape: (3242, 224, 224, 3)
Class distribution: [512 979 955 796]

Class weights: {0: 1.583, 1: 0.828, 2: 0.849, 3: 1.018}

Training set: 2269 samples
Validation set: 486 samples
Test set: 487 samples

Creating EfficientNetB0 transfer learning model...

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 efficientnetb0              (None, 7, 7, 1280)        4,049,571
 global_average_pooling2d    (None, 1280)              0         
 dense                       (None, 512)               655,872   
 batch_normalization         (None, 512)               2,048     
 dropout                     (None, 512)               0         
 dense_1                     (None, 256)               131,328   
 batch_normalization_1       (None, 256)               1,024     
 dropout_1                   (None, 256)               0         
 dense_2                     (None, 4)                 1,028     
=================================================================
Total params: 4,840,871
Trainable params: 791,300
Non-trainable params: 4,049,571
_________________________________________________________________

Training and fine-tuning completed.

Evaluating on test set...

Test Accuracy: 0.5688
Test Loss: 1.1069

Classification Report:
                         precision    recall  f1-score   support

                 Benign       0.43      0.45      0.44        77
[Malignant] early Pre-B       0.67      0.03      0.05       147
      [Malignant] Pre-B       0.70      0.83      0.76       143
      [Malignant] Pro-B       0.52      1.00      0.68       120

               accuracy                           0.57       487
              macro avg       0.58      0.58      0.48       487
           weighted avg       0.60      0.57      0.48       487

Phase 2 dashboard saved to: results/web/phase2_dashboard.html

Phase 2 completed successfully!
Models saved in: models/phase2
Results saved in: results/figures
Web dashboard saved in: results/web
```