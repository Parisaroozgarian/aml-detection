import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")


class BALLTransferLearning:
    """
    B-ALL subtype classification with EfficientNetB0.

    Important limitation:
    The public 4-class copy used here does not provide a verified patient ID
    for each image, so the split below is image-level, not patient-level.
    Exact byte-for-byte duplicate files are removed before splitting.
    """

    def __init__(self, data_path, img_size=224):
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.class_names = [
            "Benign",
            "[Malignant] early Pre-B",
            "[Malignant] Pre-B",
            "[Malignant] Pro-B",
        ]
        self.num_classes = len(self.class_names)

        self.models_dir = Path("models/phase2")
        self.results_dir = Path("results")
        self.figures_dir = self.results_dir / "figures"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sha256(path):
        """Return SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def preprocess_image(self, img):
        """
        Apply CLAHE while keeping values on the 0-255 scale.

        Keras EfficientNetB0 includes its own rescaling layer, so we do NOT
        divide by 255 here.
        """
        img_uint8 = img.astype(np.uint8)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        for channel in range(3):
            img_uint8[:, :, channel] = clahe.apply(
                img_uint8[:, :, channel]
            )

        return img_uint8.astype(np.float32)

    def load_and_preprocess_data(self):
        """Load images, remove exact duplicates, and retain file paths."""
        print("\nLoading and preprocessing data...")

        images = []
        labels = []
        file_paths = []

        # hash -> (class_idx, original_path)
        seen_hashes = {}
        duplicate_rows = []

        for class_idx, class_name in enumerate(self.class_names):
            class_path = self.data_path / class_name

            if not class_path.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_path}"
                )

            image_files = sorted(
                [
                    p
                    for p in class_path.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            )

            class_loaded = 0
            class_duplicates = 0

            for img_path in image_files:
                try:
                    file_hash = self.sha256(img_path)

                    if file_hash in seen_hashes:
                        previous_class, previous_path = seen_hashes[file_hash]

                        duplicate_rows.append(
                            {
                                "hash": file_hash,
                                "duplicate": str(img_path),
                                "original": previous_path,
                            }
                        )
                        class_duplicates += 1

                        if previous_class != class_idx:
                            print(
                                "\nWARNING: identical image found in "
                                "different classes:"
                            )
                            print(f"  {previous_path}")
                            print(f"  {img_path}")

                        # Skip exact duplicate
                        continue

                    seen_hashes[file_hash] = (
                        class_idx,
                        str(img_path),
                    )

                    img = cv2.imread(str(img_path))
                    if img is None:
                        print(f"WARNING: could not read {img_path}")
                        continue

                    img = cv2.cvtColor(
                        img,
                        cv2.COLOR_BGR2RGB,
                    )
                    img = cv2.resize(
                        img,
                        (self.img_size, self.img_size),
                    )
                    img = self.preprocess_image(img)

                    images.append(img)
                    labels.append(class_idx)
                    file_paths.append(str(img_path))
                    class_loaded += 1

                except Exception as exc:
                    print(f"Error loading {img_path}: {exc}")

            print(
                f"{class_name}: "
                f"{class_loaded} unique images loaded, "
                f"{class_duplicates} exact duplicates skipped"
            )

        if not images:
            raise ValueError(
                "No images were loaded. Check the dataset path."
            )

        X = np.asarray(images, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        file_paths = np.asarray(file_paths)

        print("\n--- Dataset summary ---")
        print(f"Unique images loaded: {len(X)}")
        print(f"Exact duplicates skipped: {len(duplicate_rows)}")
        print(f"Image shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")

        duplicate_csv = self.results_dir / "duplicates_skipped.csv"
        with open(
            duplicate_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["hash", "duplicate", "original"],
            )
            writer.writeheader()
            writer.writerows(duplicate_rows)

        print(f"Duplicate log saved to: {duplicate_csv}")

        return X, y, file_paths

    def split_data(self, X, y, file_paths):
        """
        Stratified 70/15/15 image-level split.

        Note: this is NOT patient-level because verified patient IDs are not
        available in this public 4-class copy.
        """
        (
            X_train,
            X_temp,
            y_train,
            y_temp,
            paths_train,
            paths_temp,
        ) = train_test_split(
            X,
            y,
            file_paths,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )

        (
            X_val,
            X_test,
            y_val,
            y_test,
            paths_val,
            paths_test,
        ) = train_test_split(
            X_temp,
            y_temp,
            paths_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp,
        )

        print("\n--- Split sizes ---")
        print(f"Training set: {len(X_train)}")
        print(f"Validation set: {len(X_val)}")
        print(f"Test set: {len(X_test)}")

        manifest_path = self.results_dir / "split_manifest.csv"
        with open(
            manifest_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                ["split", "file_path", "true_class"]
            )

            for path, label in zip(paths_train, y_train):
                writer.writerow(
                    [
                        "train",
                        path,
                        self.class_names[int(label)],
                    ]
                )

            for path, label in zip(paths_val, y_val):
                writer.writerow(
                    [
                        "validation",
                        path,
                        self.class_names[int(label)],
                    ]
                )

            for path, label in zip(paths_test, y_test):
                writer.writerow(
                    [
                        "test",
                        path,
                        self.class_names[int(label)],
                    ]
                )

        print(f"Split manifest saved to: {manifest_path}")

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            paths_train,
            paths_val,
            paths_test,
        )

    def create_data_generators(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=32,
    ):
        """Training augmentation; no augmentation for validation."""
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest",
        )

        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

        train_generator = train_datagen.flow(
            X_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=42,
        )

        val_generator = val_datagen.flow(
            X_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
        )

        return train_generator, val_generator

    def create_transfer_model(self):
        """Create EfficientNetB0 transfer-learning model."""
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3),
        )

        base_model.trainable = False

        model = tf.keras.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(
                    self.num_classes,
                    activation="softmax",
                ),
            ],
            name="b_all_efficientnetb0",
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-4
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model, base_model

    def fine_tune_model(self, model, base_model):
        """Unfreeze the second half of EfficientNetB0."""
        base_model.trainable = True

        fine_tune_at = len(base_model.layers) // 2

        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-5
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def make_callbacks(self, checkpoint_name):
        """Create fresh callbacks for each training stage."""
        return [
            tf.keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor="val_accuracy",
                mode="max",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.2,
                patience=8,
                min_lr=1e-8,
                monitor="val_accuracy",
                mode="max",
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(
                    self.models_dir / checkpoint_name
                ),
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
                verbose=1,
            ),
        ]

    def plot_training_history(
        self,
        initial_history,
        fine_tune_history,
    ):
        """Save training/validation accuracy and loss plots."""
        train_acc = (
            initial_history.history["accuracy"]
            + fine_tune_history.history["accuracy"]
        )
        val_acc = (
            initial_history.history["val_accuracy"]
            + fine_tune_history.history["val_accuracy"]
        )
        train_loss = (
            initial_history.history["loss"]
            + fine_tune_history.history["loss"]
        )
        val_loss = (
            initial_history.history["val_loss"]
            + fine_tune_history.history["val_loss"]
        )

        epochs = range(1, len(train_acc) + 1)

        plt.figure(figsize=(9, 6))
        plt.plot(epochs, train_acc, label="Training Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.axvline(
            x=len(initial_history.history["accuracy"]),
            linestyle="--",
            label="Fine-tuning starts",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("EfficientNetB0 Training Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "training_accuracy.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(9, 6))
        plt.plot(epochs, train_loss, label="Training Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.axvline(
            x=len(initial_history.history["loss"]),
            linestyle="--",
            label="Fine-tuning starts",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("EfficientNetB0 Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "training_loss.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def save_test_results(
        self,
        model,
        X_test,
        y_test,
        paths_test,
    ):
        """Evaluate test set and save research-ready metrics."""
        print("\nEvaluating on the frozen test set...")

        test_loss, test_accuracy = model.evaluate(
            X_test,
            y_test,
            verbose=0,
        )

        probabilities = model.predict(
            X_test,
            verbose=0,
        )
        y_pred = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)

        report_text = classification_report(
            y_test,
            y_pred,
            target_names=self.class_names,
            digits=4,
        )

        report_dict = classification_report(
            y_test,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            digits=4,
        )

        macro_f1 = report_dict["macro avg"]["f1-score"]

        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print("\nClassification Report:")
        print(report_text)

        # Save text report
        with open(
            self.results_dir / "classification_report.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report_text)

        # Save structured report
        with open(
            self.results_dir / "classification_report.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(report_dict, f, indent=2)

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 7))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names,
        )
        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
            xticks_rotation=30,
        )
        ax.set_title("B-ALL Subtype Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            self.figures_dir / "confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Save per-image predictions for later calibration/referral work
        predictions_path = (
            self.results_dir / "test_predictions.csv"
        )

        with open(
            predictions_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.writer(f)

            header = [
                "file_path",
                "true_class",
                "predicted_class",
                "confidence",
                "correct",
            ] + [
                f"prob_{name}" for name in self.class_names
            ]

            writer.writerow(header)

            for (
                path,
                true_label,
                pred_label,
                confidence,
                probs,
            ) in zip(
                paths_test,
                y_test,
                y_pred,
                confidences,
                probabilities,
            ):
                writer.writerow(
                    [
                        path,
                        self.class_names[int(true_label)],
                        self.class_names[int(pred_label)],
                        float(confidence),
                        int(true_label == pred_label),
                    ]
                    + [float(p) for p in probs]
                )

        summary = {
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "macro_f1": float(macro_f1),
            "num_test_images": int(len(y_test)),
            "split_type": "image-level stratified",
            "patient_level_split": False,
            "exact_duplicates_removed_before_split": True,
        }

        with open(
            self.results_dir / "test_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2)

        print(
            f"Test predictions saved to: {predictions_path}"
        )

        return (
            test_accuracy,
            test_loss,
            macro_f1,
            y_pred,
            probabilities,
        )

    def train_and_evaluate(self):
        """Full training and evaluation pipeline."""
        X, y, file_paths = self.load_and_preprocess_data()

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            paths_train,
            paths_val,
            paths_test,
        ) = self.split_data(X, y, file_paths)

        # Compute class weights from TRAINING DATA ONLY
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weight_dict = dict(
            zip(classes, class_weights)
        )
        print(
            f"Training class weights: {class_weight_dict}"
        )

        train_gen, val_gen = self.create_data_generators(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=32,
        )

        print("\nCreating EfficientNetB0 model...")
        model, base_model = self.create_transfer_model()
        model.summary()

        # Stage 1: frozen base
        print("\nStage 1: training frozen EfficientNetB0...")
        initial_history = model.fit(
            train_gen,
            epochs=20,
            validation_data=val_gen,
            callbacks=self.make_callbacks(
                "best_efficientnetb0_stage1.keras"
            ),
            class_weight=class_weight_dict,
            verbose=1,
        )

        # Stage 2: fine-tuning
        print("\nStage 2: fine-tuning EfficientNetB0...")
        model = self.fine_tune_model(
            model,
            base_model,
        )

        initial_epochs_completed = len(
            initial_history.epoch
        )
        fine_tune_epochs = 15
        total_epochs = (
            initial_epochs_completed + fine_tune_epochs
        )

        fine_tune_history = model.fit(
            train_gen,
            epochs=total_epochs,
            initial_epoch=initial_epochs_completed,
            validation_data=val_gen,
            callbacks=self.make_callbacks(
                "best_efficientnetb0_finetuned.keras"
            ),
            class_weight=class_weight_dict,
            verbose=1,
        )

        final_model_path = (
            self.models_dir / "efficientnetb0_final.keras"
        )
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        self.plot_training_history(
            initial_history,
            fine_tune_history,
        )

        (
            test_accuracy,
            test_loss,
            macro_f1,
            _,
            _,
        ) = self.save_test_results(
            model,
            X_test,
            y_test,
            paths_test,
        )

        return {
            "model": model,
            "initial_history": initial_history,
            "fine_tune_history": fine_tune_history,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "macro_f1": macro_f1,
        }


def main():
    print("B-ALL Subtype Classification - Research Evaluation")
    print("=" * 60)

    # Your local Kaggle dataset path
    data_path = Path(
        r"C:\Users\paris\Downloads\dataset\Blood cell Cancer [ALL]"
    )

    if not data_path.exists():
        print(f"ERROR: dataset path does not exist:\n{data_path}")
        return

    experiment = BALLTransferLearning(
        data_path=data_path,
        img_size=224,
    )

    results = experiment.train_and_evaluate()

    print("\nCompleted successfully.")
    print(
        f"Final test accuracy: "
        f"{results['test_accuracy']:.2%}"
    )
    print(
        f"Macro F1: "
        f"{results['macro_f1']:.4f}"
    )
    print("Results saved in: results/")
    print("Model saved in: models/phase2/")


if __name__ == "__main__":
    main()
