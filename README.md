# AML Detection using Deep Learning

A deep learning project for detecting Acute Myeloid Leukemia (AML) from blood cell images using Convolutional Neural Networks and Transfer Learning.

## 📋 Project Overview

This project implements a two-phase approach for AML detection:
- **Phase 1**: Basic CNN model for initial classification
- **Phase 2**: Transfer Learning with EfficientNetB0 for improved accuracy

## 🗂️ Project Structure

```
aml-detection/
├── data/                          # Dataset (see data/README.md for download)
│   ├── Blood cell Cancer [ALL]/
│   └── raw/
├── models/                        # Trained models
│   ├── phase1/
│   │   ├── basic_cnn.h5
│   │   └── best_model.h5
│   └── phase2/
│       ├── best_efficientnetb0.h5
│       └── efficientnetb0_final.h5
├── results/                       # Training results and visualizations
├── scripts/                       # Training scripts
│   ├── phase1_basic_model.py
│   └── phase2_transfer_learning.py
└── requirements.txt               # Python dependencies
```

## 📊 Dataset

**Note**: This project uses the Blood Cell Cancer [ALL] - 4 Class dataset from Kaggle.

### Download Instructions:

📥 **Kaggle Dataset**: [Blood Cell Cancer [ALL] - 4 Class](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class)

- **Size**: ~6 GB
- **Format**: Blood cell microscopy images
- **Classes**: 4 classes (Malignant early Pre-B, Pre-B, Pro-B, and Benign)

### Setup the Dataset:

**Method 1: Manual Download**
1. Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class)
2. Click "Download" (requires free Kaggle account)
3. Extract the downloaded zip file
4. Move the extracted folder to the `data/` directory

**Method 2: Using Kaggle API**
```bash
# Install Kaggle API
pip install kaggle

# Download dataset (make sure your Kaggle API credentials are configured)
kaggle datasets download -d mohammadamireshraghi/blood-cell-cancer-all-4class

# Extract to data folder
unzip blood-cell-cancer-all-4class.zip -d data/
```

For more details, see `data/README.md`

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Parisaroozgarian/aml-detection.git
cd aml-detection
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset (see Dataset section above)

## 💻 Usage

### Phase 1 - Basic CNN:
```bash
python scripts/phase1_basic_model.py
```

### Phase 2 - Transfer Learning:
```bash
python scripts/phase2_transfer_learning.py
```

## 📈 Results

Training results, visualizations, and performance metrics are available in the `results/` folder.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.


## 🙏 Acknowledgments

- Dataset source: Blood Cells Cancer (ALL) dataset. (2021, October 30). Kaggle. https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class
- Inspired by medical AI research in leukemia detection

## 📚 Citation

If you use this dataset in your research, please cite:

```
Blood Cells Cancer (ALL) dataset. (2021, October 30). Kaggle. https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class
```