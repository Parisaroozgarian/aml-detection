# # phase1_basic_model.py
# # Complete AML detection basic CNN model for 4-class classification with web dashboard

# import tensorflow as tf
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import os
# import json
# import webbrowser
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.model_selection import train_test_split
# import warnings
# warnings.filterwarnings('ignore')

# # Set random seeds for reproducibility
# np.random.seed(42)
# tf.random.set_seed(42)

# print(f"TensorFlow: {tf.__version__}")
# print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# class AMLBasicCNN:
#     def __init__(self, data_path, img_size=224):
#         self.data_path = Path(data_path)
#         self.img_size = img_size
#         self.class_names = ['Benign', '[Malignant] early Pre-B', '[Malignant] Pre-B', '[Malignant] Pro-B']
#         self.num_classes = len(self.class_names)
        
#         # Create directories for saving results
#         self.models_dir = Path('models/phase1')
#         self.results_dir = Path('results/figures')
#         self.web_dir = Path('results/web')
#         self.models_dir.mkdir(parents=True, exist_ok=True)
#         self.results_dir.mkdir(parents=True, exist_ok=True)
#         self.web_dir.mkdir(parents=True, exist_ok=True)
        
#     def load_and_preprocess_data(self):
#         """Load images and labels from the dataset"""
#         print("Loading and preprocessing data...")
        
#         images = []
#         labels = []
        
#         # Load data from each class folder
#         for class_idx, class_name in enumerate(self.class_names):
#             class_path = self.data_path / class_name
            
#             if not class_path.exists():
#                 print(f"Warning: {class_path} does not exist!")
#                 continue
                
#             print(f"Loading {class_name}...")
#             image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')) + list(class_path.glob('*.jpeg'))
            
#             for img_path in image_files:
#                 try:
#                     # Load and preprocess image
#                     img = cv2.imread(str(img_path))
#                     if img is None:
#                         continue
                        
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                     img = cv2.resize(img, (self.img_size, self.img_size))
#                     img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
                    
#                     images.append(img)
#                     labels.append(class_idx)
                    
#                 except Exception as e:
#                     print(f"Error loading {img_path}: {e}")
#                     continue
            
#             print(f"Loaded {len([l for l in labels if l == class_idx])} images for {class_name}")
        
#         if len(images) == 0:
#             raise ValueError("No images were loaded! Check your data path.")
            
#         # Convert to numpy arrays
#         X = np.array(images)
#         y = np.array(labels)
        
#         print(f"Total images loaded: {len(X)}")
#         print(f"Image shape: {X.shape}")
#         print(f"Class distribution: {np.bincount(y)}")
        
#         return X, y
    
#     def create_model(self):
#         """Create the basic CNN model"""
#         model = tf.keras.Sequential([
#             # First Conv Block
#             tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.MaxPooling2D(2, 2),
#             tf.keras.layers.Dropout(0.25),
            
#             # Second Conv Block
#             tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.MaxPooling2D(2, 2),
#             tf.keras.layers.Dropout(0.25),
            
#             # Third Conv Block
#             tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.MaxPooling2D(2, 2),
#             tf.keras.layers.Dropout(0.25),
            
#             # Classifier
#             tf.keras.layers.GlobalAveragePooling2D(),
#             tf.keras.layers.Dense(512, activation='relu'),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Dropout(0.5),
#             tf.keras.layers.Dense(self.num_classes, activation='softmax')
#         ])
        
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy']
#         )
        
#         return model
    
#     def plot_training_history(self, history):
#         """Plot training history"""
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
#         # Plot accuracy
#         ax1.plot(history.history['accuracy'], label='Training Accuracy')
#         ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
#         ax1.set_title('Model Accuracy')
#         ax1.set_xlabel('Epoch')
#         ax1.set_ylabel('Accuracy')
#         ax1.legend()
#         ax1.grid(True)
        
#         # Plot loss
#         ax2.plot(history.history['loss'], label='Training Loss')
#         ax2.plot(history.history['val_loss'], label='Validation Loss')
#         ax2.set_title('Model Loss')
#         ax2.set_xlabel('Epoch')
#         ax2.set_ylabel('Loss')
#         ax2.legend()
#         ax2.grid(True)
        
#         plt.tight_layout()
#         plt.savefig(self.results_dir / 'training_history.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def plot_predictions(self, model, X_test, y_test, num_samples=16):
#         """Plot sample predictions"""
#         predictions = model.predict(X_test)
#         predicted_classes = np.argmax(predictions, axis=1)
        
#         # Select random samples
#         indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        
#         fig, axes = plt.subplots(4, 4, figsize=(12, 12))
#         axes = axes.ravel()
        
#         for i, idx in enumerate(indices):
#             if i >= 16:
#                 break
                
#             axes[i].imshow(X_test[idx])
#             true_label = self.class_names[y_test[idx]]
#             pred_label = self.class_names[predicted_classes[idx]]
#             confidence = np.max(predictions[idx]) * 100
            
#             color = 'green' if y_test[idx] == predicted_classes[idx] else 'red'
#             axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', 
#                             color=color, fontsize=8)
#             axes[i].axis('off')
        
#         plt.tight_layout()
#         plt.savefig(self.results_dir / 'predictions.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def plot_confusion_matrix(self, y_true, y_pred):
#         """Plot confusion matrix"""
#         cm = confusion_matrix(y_true, y_pred)
        
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                    xticklabels=self.class_names, yticklabels=self.class_names)
#         plt.title('Confusion Matrix')
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.xticks(rotation=45)
#         plt.yticks(rotation=0)
#         plt.tight_layout()
#         plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def generate_web_dashboard(self, results_data):
#         """Generate interactive web dashboard"""
#         dashboard_html = f"""<!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>AML Detection Results Dashboard</title>
#     <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
#     <style>
#         * {{
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }}

#         body {{
#             font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: #333;
#             line-height: 1.6;
#         }}

#         .container {{
#             max-width: 1400px;
#             margin: 0 auto;
#             padding: 20px;
#         }}

#         .header {{
#             text-align: center;
#             color: white;
#             margin-bottom: 30px;
#             padding: 20px;
#             background: rgba(255, 255, 255, 0.1);
#             border-radius: 15px;
#             backdrop-filter: blur(10px);
#         }}

#         .header h1 {{
#             font-size: 2.5rem;
#             margin-bottom: 10px;
#             font-weight: 700;
#         }}

#         .dashboard-grid {{
#             display: grid;
#             grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
#             gap: 20px;
#             margin-bottom: 30px;
#         }}

#         .card {{
#             background: white;
#             border-radius: 15px;
#             padding: 25px;
#             box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
#             transition: transform 0.3s ease;
#         }}

#         .card:hover {{
#             transform: translateY(-5px);
#         }}

#         .card h3 {{
#             color: #4a5568;
#             margin-bottom: 20px;
#             font-size: 1.3rem;
#             font-weight: 600;
#             border-bottom: 2px solid #e2e8f0;
#             padding-bottom: 10px;
#         }}

#         .metrics-grid {{
#             display: grid;
#             grid-template-columns: repeat(2, 1fr);
#             gap: 15px;
#             margin-bottom: 20px;
#         }}

#         .metric {{
#             text-align: center;
#             padding: 15px;
#             background: linear-gradient(135deg, #667eea, #764ba2);
#             color: white;
#             border-radius: 10px;
#         }}

#         .metric-value {{
#             font-size: 2rem;
#             font-weight: bold;
#             margin-bottom: 5px;
#         }}

#         .chart-container {{
#             position: relative;
#             height: 300px;
#             margin: 20px 0;
#         }}

#         .full-width {{
#             grid-column: 1 / -1;
#         }}
#     </style>
# </head>
# <body>
#     <div class="container">
#         <div class="header">
#             <h1>🩸 AML Detection Model Dashboard</h1>
#             <p>Deep Learning for Acute Myeloid Leukemia Classification</p>
#         </div>

#         <div class="dashboard-grid">
#             <div class="card">
#                 <h3>📊 Model Performance</h3>
#                 <div class="metrics-grid">
#                     <div class="metric">
#                         <div class="metric-value">{results_data['test_accuracy']:.1%}</div>
#                         <div class="metric-label">Test Accuracy</div>
#                     </div>
#                     <div class="metric">
#                         <div class="metric-value">{results_data['test_loss']:.3f}</div>
#                         <div class="metric-label">Test Loss</div>
#                     </div>
#                     <div class="metric">
#                         <div class="metric-value">{results_data['precision']:.1%}</div>
#                         <div class="metric-label">Precision</div>
#                     </div>
#                     <div class="metric">
#                         <div class="metric-value">{results_data['recall']:.1%}</div>
#                         <div class="metric-label">Recall</div>
#                     </div>
#                 </div>
#             </div>

#             <div class="card">
#                 <h3>📈 Training History</h3>
#                 <div class="chart-container">
#                     <canvas id="trainingChart"></canvas>
#                 </div>
#             </div>

#             <div class="card full-width">
#                 <h3>🧬 Class Distribution</h3>
#                 <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
#                     {' '.join([f'<div style="text-align: center; padding: 15px; background: #f7fafc; border-radius: 8px;"><div style="font-weight: 600; margin-bottom: 5px;">{name}</div><div style="font-size: 1.5rem; color: #4299e1; font-weight: bold;">{count}</div></div>' for name, count in zip(results_data['class_names'], results_data['class_counts'])])}
#                 </div>
#             </div>
#         </div>
#     </div>

#     <script>
#         const trainingCtx = document.getElementById('trainingChart').getContext('2d');
#         new Chart(trainingCtx, {{
#             type: 'line',
#             data: {{
#                 labels: {list(range(1, len(results_data['training_accuracy']) + 1))},
#                 datasets: [{{
#                     label: 'Training Accuracy',
#                     data: {results_data['training_accuracy']},
#                     borderColor: '#4299e1',
#                     backgroundColor: 'rgba(66, 153, 225, 0.1)',
#                     tension: 0.4
#                 }}, {{
#                     label: 'Validation Accuracy',
#                     data: {results_data['validation_accuracy']},
#                     borderColor: '#48bb78',
#                     backgroundColor: 'rgba(72, 187, 120, 0.1)',
#                     tension: 0.4
#                 }}]
#             }},
#             options: {{
#                 responsive: true,
#                 maintainAspectRatio: false,
#                 plugins: {{
#                     legend: {{
#                         position: 'top'
#                     }}
#                 }},
#                 scales: {{
#                     y: {{
#                         beginAtZero: false,
#                         min: 0.4,
#                         max: 1.0
#                     }}
#                 }}
#             }}
#         }});
#     </script>
# </body>
# </html>"""
        
#         # Save dashboard
#         dashboard_path = self.web_dir / 'dashboard.html'
#         with open(dashboard_path, 'w') as f:
#             f.write(dashboard_html)
        
#         print(f"Web dashboard saved to: {dashboard_path}")
        
#         # Open dashboard in browser
#         try:
#             webbrowser.open(f'file://{dashboard_path.absolute()}')
#             print("Dashboard opened in your default browser!")
#         except:
#             print(f"Please open {dashboard_path} in your browser to view the dashboard.")
    
#     def train_and_evaluate(self):
#         """Main training and evaluation pipeline"""
#         # Load data
#         X, y = self.load_and_preprocess_data()
        
#         # Split data
#         X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
#         X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
#         print(f"Training set: {len(X_train)} samples")
#         print(f"Validation set: {len(X_val)} samples")
#         print(f"Test set: {len(X_test)} samples")
        
#         # Create model
#         model = self.create_model()
#         model.summary()
        
#         # Callbacks
#         callbacks = [
#             tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
#             tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-7),
#             tf.keras.callbacks.ModelCheckpoint(
#                 str(self.models_dir / 'best_model.h5'),  # Convert to string
#                 save_best_only=True,
#                 monitor='val_accuracy'
#             )
#         ]
        
#         # Train model
#         print("\nStarting training...")
#         history = model.fit(
#             X_train, y_train,
#             epochs=50,
#             batch_size=32,
#             validation_data=(X_val, y_val),
#             callbacks=callbacks,
#             verbose=1
#         )
        
#         # Save final model
#         model.save(str(self.models_dir / 'basic_cnn.h5'))  # Convert to string
#         print(f"Model saved to {self.models_dir / 'basic_cnn.h5'}")
        
#         # Evaluate on test set
#         print("\nEvaluating on test set...")
#         test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
#         print(f"Test Accuracy: {test_accuracy:.4f}")
#         print(f"Test Loss: {test_loss:.4f}")
        
#         # Predictions
#         y_pred = np.argmax(model.predict(X_test), axis=1)
        
#         # Classification report
#         report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
#         print("\nClassification Report:")
#         print(classification_report(y_test, y_pred, target_names=self.class_names))
        
#         # Prepare data for web dashboard
#         results_data = {
#             'test_accuracy': test_accuracy,
#             'test_loss': test_loss,
#             'precision': report['macro avg']['precision'],
#             'recall': report['macro avg']['recall'],
#             'training_accuracy': history.history['accuracy'],
#             'validation_accuracy': history.history['val_accuracy'],
#             'class_names': self.class_names,
#             'class_counts': np.bincount(y).tolist()
#         }
        
#         # Generate web dashboard
#         self.generate_web_dashboard(results_data)
        
#         # Plot traditional results
#         self.plot_training_history(history)
#         self.plot_predictions(model, X_test, y_test)
#         self.plot_confusion_matrix(y_test, y_pred)
        
#         return model, history

# def main():
#     """Main execution function"""
#     print("Phase 1: Basic CNN Model for AML Detection")
#     print("=" * 50)
    
#     # Set data path - adjust this to match your structure
#     data_path = Path("data/Blood cell Cancer [ALL]")
    
#     if not data_path.exists():
#         print(f"Error: Data path {data_path} does not exist!")
#         print("Please check your data path and update the script accordingly.")
#         return
    
#     # Create and train model
#     aml_cnn = AMLBasicCNN(data_path)
#     model, history = aml_cnn.train_and_evaluate()
    
#     print("\nTraining completed successfully!")
#     print(f"Models saved in: {aml_cnn.models_dir}")
#     print(f"Results saved in: {aml_cnn.results_dir}")
#     print(f"Web dashboard saved in: {aml_cnn.web_dir}")

# if __name__ == "__main__":
#     main()
    
    
# phase2_transfer_learning.py
# Advanced AML detection using Transfer Learning with pre-trained models
# Expected accuracy: 85-90% (improvement from 70-80% in Phase 1)

# phase2_transfer_learning.py
# Advanced AML detection using Transfer Learning with pre-trained models
# Expected accuracy: 85-90% (improvement from 70-80% in Phase 1)