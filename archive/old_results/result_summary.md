# AML Detection Model Results – Phase 1 (Basic CNN)

## Model Performance
- **Test Accuracy:** 0.9897 (98.97%)
- **Test Loss:** 0.0518
- **Model:** Custom CNN Architecture

## Classification Report
```
                         precision    recall  f1-score   support

                 Benign       0.99      0.96      0.97        77
[Malignant] early Pre-B       0.99      1.00      0.99       147
      [Malignant] Pre-B       1.00      0.99      1.00       143
      [Malignant] Pro-B       0.98      0.99      0.99       120

               accuracy                           0.99       487
              macro avg       0.99      0.99      0.99       487
           weighted avg       0.99      0.99      0.99       487
```

## Figures
- Confusion Matrix: `results/figures/confusion_matrix.png`
- Predictions: `results/figures/predictions.png`
- Training History: `results/figures/training_history.png`

## Web Dashboard
- Dashboard: `results/web/dashboard.html`

---

Phase 1 completed successfully!  
Models saved in: `models/phase1`  
Results saved in: `results/figures`  
Web dashboard saved in: `results/web`

---

## Training Log

```
TensorFlow: 2.13.0
GPU Available: False
Phase 1: Basic CNN Model for AML Detection
==================================================

Loading and preprocessing data...
Loaded 512 images for Benign
Loaded 979 images for [Malignant] early Pre-B
Loaded 955 images for [Malignant] Pre-B
Loaded 796 images for [Malignant] Pro-B
Total images loaded: 3242
Image shape: (3242, 224, 224, 3)
Class distribution: [512 979 955 796]

Training set: 2269 samples
Validation set: 486 samples
Test set: 487 samples

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 222, 222, 32)      896       
 batch_normalization         (None, 222, 222, 32)      128       
 max_pooling2d               (None, 111, 111, 32)      0         
 dropout                     (None, 111, 111, 32)      0         
 conv2d_1 (Conv2D)           (None, 109, 109, 64)      18496     
 batch_normalization_1       (None, 109, 109, 64)      256       
 max_pooling2d_1             (None, 54, 54, 64)        0         
 dropout_1                   (None, 54, 54, 64)        0         
 conv2d_2 (Conv2D)           (None, 52, 52, 128)       73856     
 batch_normalization_2       (None, 52, 52, 128)       512       
 max_pooling2d_2             (None, 26, 26, 128)       0         
 dropout_2                   (None, 26, 26, 128)       0         
 global_average_pooling2d    (None, 128)               0         
 dense (Dense)               (None, 512)               66048     
 batch_normalization_3       (None, 512)               2048      
 dropout_3                   (None, 512)               0         
 dense_1 (Dense)             (None, 4)                 2052      
=================================================================
Total params: 164292 (641.77 KB)
Trainable params: 162820 (636.02 KB)
Non-trainable params: 1472 (5.75 KB)
_________________________________________________________________

Starting training...
Epoch 1/50
71/71 [==============================] - 529s 7s/step - loss: 0.3767 - accuracy: 0.8770 - val_loss: 1.8777 - val_accuracy: 0.3251 - lr: 0.0010

Epoch 2/50
71/71 [==============================] - 475s 7s/step - loss: 0.1490 - accuracy: 0.9520 - val_loss: 3.3084 - val_accuracy: 0.3025 - lr: 0.0010

Epoch 3/50
71/71 [==============================] - 531s 8s/step - loss: 0.1371 - accuracy: 0.9564 - val_loss: 4.9390 - val_accuracy: 0.3025 - lr: 0.0010

Epoch 4/50
71/71 [==============================] - 456s 6s/step - loss: 0.0994 - accuracy: 0.9722 - val_loss: 4.3062 - val_accuracy: 0.3724 - lr: 0.0010

Epoch 5/50
71/71 [==============================] - 469s 7s/step - loss: 0.0937 - accuracy: 0.9709 - val_loss: 8.7139 - val_accuracy: 0.2510 - lr: 0.0010

Epoch 6/50
71/71 [==============================] - 483s 7s/step - loss: 0.0898 - accuracy: 0.9678 - val_loss: 9.5246 - val_accuracy: 0.2449 - lr: 0.0010

Epoch 7/50
71/71 [==============================] - 480s 7s/step - loss: 0.0673 - accuracy: 0.9802 - val_loss: 5.0303 - val_accuracy: 0.4918 - lr: 2.0000e-04

Epoch 8/50
71/71 [==============================] - 435s 6s/step - loss: 0.0370 - accuracy: 0.9881 - val_loss: 4.9880 - val_accuracy: 0.3210 - lr: 2.0000e-04

Epoch 9/50
71/71 [==============================] - 446s 6s/step - loss: 0.0309 - accuracy: 0.9903 - val_loss: 1.6164 - val_accuracy: 0.7490 - lr: 2.0000e-04

Epoch 10/50
71/71 [==============================] - 451s 6s/step - loss: 0.0201 - accuracy: 0.9943 - val_loss: 2.3975 - val_accuracy: 0.6193 - lr: 2.0000e-04

Epoch 11/50
71/71 [==============================] - 559s 8s/step - loss: 0.0288 - accuracy: 0.9907 - val_loss: 1.6299 - val_accuracy: 0.8313 - lr: 2.0000e-04

Epoch 12/50
71/71 [==============================] - 450s 6s/step - loss: 0.0243 - accuracy: 0.9925 - val_loss: 0.6300 - val_accuracy: 0.8951 - lr: 2.0000e-04

Epoch 13/50
71/71 [==============================] - 452s 6s/step - loss: 0.0250 - accuracy: 0.9921 - val_loss: 1.0940 - val_accuracy: 0.8560 - lr: 2.0000e-04

Epoch 14/50
71/71 [==============================] - 433s 6s/step - loss: 0.0251 - accuracy: 0.9929 - val_loss: 1.7889 - val_accuracy: 0.8354 - lr: 2.0000e-04

Epoch 15/50
71/71 [==============================] - 446s 6s/step - loss: 0.0231 - accuracy: 0.9925 - val_loss: 2.2164 - val_accuracy: 0.7119 - lr: 2.0000e-04

Epoch 16/50
71/71 [==============================] - 812s 12s/step - loss: 0.0195 - accuracy: 0.9943 - val_loss: 0.9668 - val_accuracy: 0.8621 - lr: 2.0000e-04

Epoch 17/50
71/71 [==============================] - 634s 9s/step - loss: 0.0182 - accuracy: 0.9947 - val_loss: 0.5345 - val_accuracy: 0.9136 - lr: 2.0000e-04

Epoch 18/50
71/71 [==============================] - 534s 8s/step - loss: 0.0200 - accuracy: 0.9925 - val_loss: 0.9677 - val_accuracy: 0.8045 - lr: 2.0000e-04

Epoch 19/50
71/71 [==============================] - 491s 7s/step - loss: 0.0134 - accuracy: 0.9960 - val_loss: 0.8797 - val_accuracy: 0.7778 - lr: 2.0000e-04

Epoch 20/50
71/71 [==============================] - 461s 6s/step - loss: 0.0219 - accuracy: 0.9934 - val_loss: 0.0931 - val_accuracy: 0.9835 - lr: 2.0000e-04

Epoch 21/50
71/71 [==============================] - 450s 6s/step - loss: 0.0176 - accuracy: 0.9943 - val_loss: 0.4655 - val_accuracy: 0.8930 - lr: 2.0000e-04

Epoch 22/50
71/71 [==============================] - 467s 7s/step - loss: 0.0137 - accuracy: 0.9952 - val_loss: 0.5059 - val_accuracy: 0.8580 - lr: 2.0000e-04

Epoch 23/50
71/71 [==============================] - 436s 6s/step - loss: 0.0135 - accuracy: 0.9960 - val_loss: 0.7292 - val_accuracy: 0.9136 - lr: 2.0000e-04

Epoch 24/50
71/71 [==============================] - 926s 13s/step - loss: 0.0103 - accuracy: 0.9969 - val_loss: 0.6233 - val_accuracy: 0.9033 - lr: 2.0000e-04

Epoch 25/50
71/71 [==============================] - 436s 6s/step - loss: 0.0155 - accuracy: 0.9952 - val_loss: 2.5647 - val_accuracy: 0.7243 - lr: 2.0000e-04

Epoch 26/50
71/71 [==============================] - 432s 6s/step - loss: 0.0131 - accuracy: 0.9965 - val_loss: 0.7336 - val_accuracy: 0.8889 - lr: 4.0000e-05

Epoch 27/50
71/71 [==============================] - 431s 6s/step - loss: 0.0089 - accuracy: 0.9987 - val_loss: 0.8909 - val_accuracy: 0.8786 - lr: 4.0000e-05

Epoch 28/50
71/71 [==============================] - 430s 6s/step - loss: 0.0123 - accuracy: 0.9960 - val_loss: 0.8056 - val_accuracy: 0.8848 - lr: 4.0000e-05

Epoch 29/50
71/71 [==============================] - 431s 6s/step - loss: 0.0114 - accuracy: 0.9956 - val_loss: 0.4420 - val_accuracy: 0.9280 - lr: 4.0000e-05

Epoch 30/50
71/71 [==============================] - 513s 7s/step - loss: 0.0077 - accuracy: 0.9991 - val_loss: 0.3095 - val_accuracy: 0.9403 - lr: 4.0000e-05

Model saved to models/phase1/basic_cnn.h5

Evaluating on test set...
16/16 [==============================] - 28s 2s/step

Test Accuracy: 0.9897
Test Loss: 0.0518

Classification Report:
                         precision    recall  f1-score   support

                 Benign       0.99      0.96      0.97        77
[Malignant] early Pre-B       0.99      1.00      0.99       147
      [Malignant] Pre-B       1.00      0.99      1.00       143
      [Malignant] Pro-B       0.98      0.99      0.99       120

               accuracy                           0.99       487
              macro avg       0.99      0.99      0.99       487
           weighted avg       0.99      0.99      0.99       487

Web dashboard saved to: results/web/dashboard.html
Dashboard opened in your default browser!

Training completed successfully!
Models saved in: models/phase1
Results saved in: results/figures
Web dashboard saved in: results/web
```