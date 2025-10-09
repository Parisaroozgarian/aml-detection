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

Due to GitHub file size limits, the dataset is hosted externally.

### Download Instructions:
1. **Google Drive**: [Download Dataset](YOUR_GOOGLE_DRIVE_LINK_HERE)
2. **Alternative**: [Kaggle Dataset](YOUR_KAGGLE_LINK_HERE)
3. **Size**: ~6 GB
4. **Format**: Blood cell microscopy images

### After downloading:
```bash
# Extract the dataset to the data/ folder
unzip dataset.zip -d data/
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

## 👤 Author

**Parisa Roozgarian**
- GitHub: [@Parisaroozgarian](https://github.com/Parisaroozgarian)

## 🙏 Acknowledgments

- Dataset source: [Add source here]
- Inspired by medical AI research in leukemia detection

---

## 📥 Setting Up the Dataset

After downloading the dataset from the links above:

1. Upload your data folder to **Google Drive**:
   ```bash
   # Zip your data folder first
   zip -r aml-data.zip data/
   # Upload to Google Drive and get shareable link
   ```

2. Or upload to **Kaggle** (recommended for ML projects):
   - Go to [kaggle.com](https://kaggle.com) → Create → New Dataset
   - Upload your data and get the dataset link

3. Update the README with your actual links:
   - Replace `YOUR_GOOGLE_DRIVE_LINK_HERE` with your Google Drive link
   - Replace `YOUR_KAGGLE_LINK_HERE` with your Kaggle dataset link