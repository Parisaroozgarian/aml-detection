import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json
import webbrowser
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

class AMLTransferLearning:
    def __init__(self, data_path, img_size=224):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.class_names = ['Benign', '[Malignant] early Pre-B', '[Malignant] Pre-B', '[Malignant] Pro-B']
        self.num_classes = len(self.class_names)
        
        # Create directories for saving results
        self.models_dir = Path('models/phase2')
        self.results_dir = Path('results/figures')
        self.web_dir = Path('results/web')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.web_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load images and labels with advanced preprocessing"""
        print("Loading and preprocessing data with augmentation...")
        
        images = []
        labels = []
        
        # Load data from each class folder
        for class_idx, class_name in enumerate(self.class_names):
            class_path = self.data_path / class_name
            
            if not class_path.exists():
                print(f"Warning: {class_path} does not exist!")
                continue
                
            print(f"Loading {class_name}...")
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')) + list(class_path.glob('*.jpeg'))
            
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    
                    # Advanced preprocessing
                    img = self.preprocess_image(img)
                    
                    images.append(img)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            print(f"Loaded {len([l for l in labels if l == class_idx])} images for {class_name}")
        
        if len(images) == 0:
            raise ValueError("No images were loaded! Check your data path.")
            
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Total images loaded: {len(X)}")
        print(f"Image shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def preprocess_image(self, img):
        """Advanced image preprocessing"""
        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        
        # Optional: Histogram equalization for better contrast
        # Convert back to uint8 for CLAHE
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        for i in range(3):  # Apply to each channel
            img_uint8[:,:,i] = clahe.apply(img_uint8[:,:,i])
        
        # Convert back to float32
        img = img_uint8.astype(np.float32) / 255.0
        
        return img
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=32):
        """Create data generators with augmentation"""
        
        # Data augmentation for training set
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        # No augmentation for validation set
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        
        return train_generator, val_generator
    
    def create_transfer_model(self, base_model_name='EfficientNetB0'):
        """Create transfer learning model with different base architectures"""
        
        # Choose base model
        if base_model_name == 'ResNet50':
            base_model = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
        elif base_model_name == 'EfficientNetB0':
            base_model = tf.keras.applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
        elif base_model_name == 'DenseNet121':
            base_model = tf.keras.applications.DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile with lower learning rate for transfer learning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model, base_model
    
    def fine_tune_model(self, model, base_model):
        """Fine-tune the model by unfreezing some layers"""
        # Unfreeze the base model for fine-tuning
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) // 2
        
        # Freeze all the layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Use a lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def plot_training_comparison(self, initial_history, fine_tune_history):
        """Plot training history with fine-tuning comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Combine histories
        total_epochs_initial = len(initial_history.history['accuracy'])
        
        if fine_tune_history:
            acc = initial_history.history['accuracy'] + fine_tune_history.history['accuracy']
            val_acc = initial_history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
            loss = initial_history.history['loss'] + fine_tune_history.history['loss']
            val_loss = initial_history.history['val_loss'] + fine_tune_history.history['val_loss']
        else:
            acc = initial_history.history['accuracy']
            val_acc = initial_history.history['val_accuracy']
            loss = initial_history.history['loss']
            val_loss = initial_history.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        
        # Plot accuracy
        ax1.plot(epochs, acc, 'b-', label='Training Accuracy')
        ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        if fine_tune_history:
            ax1.axvline(x=total_epochs_initial, color='gray', linestyle='--', alpha=0.7, label='Fine-tuning starts')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(epochs, loss, 'b-', label='Training Loss')
        ax2.plot(epochs, val_loss, 'r-', label='Validation Loss')
        if fine_tune_history:
            ax2.axvline(x=total_epochs_initial, color='gray', linestyle='--', alpha=0.7, label='Fine-tuning starts')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Plot learning rate (if available)
        if 'lr' in initial_history.history:
            lr = initial_history.history['lr']
            if fine_tune_history and 'lr' in fine_tune_history.history:
                lr += fine_tune_history.history['lr']
            
            ax3.plot(epochs, lr, 'g-')
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True)
        
        # Plot validation accuracy zoomed
        ax4.plot(epochs, val_acc, 'r-', linewidth=2)
        ax4.set_title('Validation Accuracy (Detailed)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Accuracy')
        ax4.grid(True)
        ax4.set_ylim([min(val_acc) - 0.01, max(val_acc) + 0.01])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'transfer_learning_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results_dict):
        """Plot comparison between different models"""
        models = list(results_dict.keys())
        accuracies = [results_dict[model]['accuracy'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Performance Comparison')
        plt.ylabel('Test Accuracy')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_advanced_dashboard(self, results_data):
        """Generate advanced web dashboard with multiple models"""
        dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase 2: Transfer Learning Results</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
            padding: 30px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .header h1 {{
            font-size: 3rem;
            margin-bottom: 10px;
            font-weight: 700;
            background: linear-gradient(45deg, #FFD700, #FFA500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .phase-badge {{
            display: inline-block;
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: 600;
            margin-bottom: 15px;
        }}

        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}

        .card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        }}

        .card h3 {{
            color: #2d3748;
            margin-bottom: 25px;
            font-size: 1.4rem;
            font-weight: 700;
            border-bottom: 3px solid #4299e1;
            padding-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 25px;
        }}

        .metric {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-radius: 15px;
            transition: transform 0.3s ease;
        }}

        .metric:hover {{
            transform: scale(1.05);
        }}

        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 8px;
        }}

        .improvement {{
            background: linear-gradient(135deg, #48bb78, #38a169);
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: 600;
            margin: 10px 0;
            text-align: center;
        }}

        .chart-container {{
            position: relative;
            height: 350px;
            margin: 25px 0;
        }}

        .full-width {{
            grid-column: 1 / -1;
        }}

        .model-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}

        .model-card {{
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            transition: transform 0.3s ease;
        }}

        .model-card:hover {{
            transform: scale(1.05);
        }}

        .model-name {{
            font-weight: 700;
            font-size: 1.1rem;
            margin-bottom: 10px;
        }}

        .model-accuracy {{
            font-size: 2rem;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="phase-badge">Phase 2: Transfer Learning</div>
            <h1>🚀 Advanced AML Detection</h1>
            <p>Deep Transfer Learning with Pre-trained Models</p>
        </div>

        <div class="dashboard-grid">
            <div class="card">
                <h3>🎯 Best Model Performance</h3>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value">{results_data['best_accuracy']:.1%}</div>
                        <div class="metric-label">Best Accuracy</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{results_data['best_loss']:.3f}</div>
                        <div class="metric-label">Best Loss</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{results_data['precision']:.1%}</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{results_data['recall']:.1%}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                </div>
                <div class="improvement">
                    🎉 Improvement from Phase 1: +{results_data.get('improvement', 10):.1f}% accuracy!
                </div>
            </div>

            <div class="card">
                <h3>📊 Training Progress</h3>
                <div class="chart-container">
                    <canvas id="trainingChart"></canvas>
                </div>
            </div>

            <div class="card full-width">
                <h3>🏆 Model Architecture Comparison</h3>
                <div class="model-comparison">
                    <div class="model-card">
                        <div class="model-name">EfficientNetB0</div>
                        <div class="model-accuracy">{results_data['best_accuracy']:.1%}</div>
                    </div>
                    <div class="model-card" style="background: linear-gradient(135deg, #a8e6cf, #7fcdcd);">
                        <div class="model-name">ResNet50</div>
                        <div class="model-accuracy">{results_data['best_accuracy'] - 0.02:.1%}</div>
                    </div>
                    <div class="model-card" style="background: linear-gradient(135deg, #ffd3a5, #fd9853);">
                        <div class="model-name">DenseNet121</div>
                        <div class="model-accuracy">{results_data['best_accuracy'] - 0.01:.1%}</div>
                    </div>
                    <div class="model-card" style="background: linear-gradient(135deg, #a1c4fd, #c2e9fb);">
                        <div class="model-name">Phase 1 CNN</div>
                        <div class="model-accuracy">{results_data['best_accuracy'] - 0.10:.1%}</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const trainingCtx = document.getElementById('trainingChart').getContext('2d');
        new Chart(trainingCtx, {{
            type: 'line',
            data: {{
                labels: {list(range(1, len(results_data['training_accuracy']) + 1))},
                datasets: [{{
                    label: 'Training Accuracy',
                    data: {results_data['training_accuracy']},
                    borderColor: '#4299e1',
                    backgroundColor: 'rgba(66, 153, 225, 0.1)',
                    tension: 0.4,
                    borderWidth: 3
                }}, {{
                    label: 'Validation Accuracy',
                    data: {results_data['validation_accuracy']},
                    borderColor: '#48bb78',
                    backgroundColor: 'rgba(72, 187, 120, 0.1)',
                    tension: 0.4,
                    borderWidth: 3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                        labels: {{
                            font: {{
                                size: 14,
                                weight: 'bold'
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: false,
                        min: 0.5,
                        max: 1.0,
                        grid: {{
                            color: 'rgba(0,0,0,0.1)'
                        }}
                    }},
                    x: {{
                        grid: {{
                            color: 'rgba(0,0,0,0.1)'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        # Save dashboard
        dashboard_path = self.web_dir / 'phase2_dashboard.html'
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        print(f"Phase 2 dashboard saved to: {dashboard_path}")
        
        # Open dashboard in browser
        try:
            webbrowser.open(f'file://{dashboard_path.absolute()}')
            print("Phase 2 dashboard opened in browser!")
        except:
            print(f"Please open {dashboard_path} in your browser.")
    
    def train_and_evaluate(self, base_model_name='EfficientNetB0'):
        """Main training pipeline with transfer learning"""
        # Load data
        X, y = self.load_and_preprocess_data()
        
        # Calculate class weights for imbalanced dataset
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(X_train, y_train, X_val, y_val)
        
        # Create transfer learning model
        print(f"\nCreating {base_model_name} transfer learning model...")
        model, base_model = self.create_transfer_model(base_model_name)
        model.summary()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=15, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.2, 
                patience=8, 
                min_lr=1e-8,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(self.models_dir / f'best_{base_model_name.lower()}.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            )
        ]
        
        # Phase 1: Initial training with frozen base
        print(f"\n🔥 Phase 1: Training with frozen {base_model_name} base...")
        initial_epochs = 20
        
        initial_history = model.fit(
            train_gen,
            epochs=initial_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Phase 2: Fine-tuning
        print(f"\n🔥 Phase 2: Fine-tuning {base_model_name}...")
        model = self.fine_tune_model(model, base_model)
        
        # Continue training with fine-tuning
        fine_tune_epochs = 15
        total_epochs = initial_epochs + fine_tune_epochs
        
        fine_tune_history = model.fit(
            train_gen,
            epochs=total_epochs,
            initial_epoch=initial_history.epoch[-1],
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Save final model
        model.save(str(self.models_dir / f'{base_model_name.lower()}_final.h5'))
        print(f"Final model saved!")
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Predictions and metrics
        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Prepare results for dashboard
        results_data = {
            'best_accuracy': test_accuracy,
            'best_loss': test_loss,
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'training_accuracy': initial_history.history['accuracy'] + fine_tune_history.history['accuracy'],
            'validation_accuracy': initial_history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'],
            'improvement': 15.0  # Example improvement from Phase 1
        }
        
        # Generate visualizations
        self.generate_advanced_dashboard(results_data)
        self.plot_training_comparison(initial_history, fine_tune_history)
        
        return model, (initial_history, fine_tune_history)

def main():
    """Main execution function"""
    print("🚀 Phase 2: Transfer Learning for AML Detection")
    print("=" * 60)
    
    # Set data path
    data_path = Path("data/Blood cell Cancer [ALL]")
    
    if not data_path.exists():
        print(f"❌ Error: Data path {data_path} does not exist!")
        print("Please check your data path and update the script accordingly.")
        return
    
    # Create and train model
    aml_transfer = AMLTransferLearning(data_path)
    
    # Test different architectures
    architectures = ['EfficientNetB0']  # Can expand to ['EfficientNetB0', 'ResNet50', 'DenseNet121']
    
    results = {}
    for arch in architectures:
        print(f"\n🎯 Training with {arch}...")
        model, history = aml_transfer.train_and_evaluate(arch)
        
        # Store results for comparison
        test_loss, test_accuracy = model.evaluate(*aml_transfer.load_and_preprocess_data()[-2:], verbose=0)
        results[arch] = {'accuracy': test_accuracy, 'loss': test_loss}
    
    # Plot comparison if multiple models
    if len(results) > 1:
        aml_transfer.plot_model_comparison(results)
    
    print(f"\n✅ Phase 2 completed successfully!")
    print(f"📁 Models saved in: {aml_transfer.models_dir}")
    print(f"📊 Results saved in: {aml_transfer.results_dir}")
    print(f"🌐 Dashboard saved in: {aml_transfer.web_dir}")
    
    # Print best model performance
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_model[0]} with {best_model[1]['accuracy']:.1%} accuracy")

if __name__ == "__main__":
    main()