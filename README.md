# AUTOMATED-PNEUMONIA-DETECTION-FROM-CHEST-X-RAY-IMAGES
Develop an automated and explainable deep learning system that accurately detects pneumonia from chest X-ray images by combining CNN-based spatial feature extraction with RNN-based sequential feature learning.
# An Explainable Hybrid ConvNeXt–LSTM Framework for Automated Pneumonia Detection from Chest X-Ray Images Using Grad-CAM

Architecture
Chest X-Ray [224×224]
        │
        ▼
ConvNeXt-Tiny (ImageNet pretrained)  →  [768, 7, 7]
        │
        ▼
Reshape to sequence [49 × 768]
        │
        ▼
Stacked LSTM (2 layers, hidden=256)
        │
        ▼
Linear classifier  →  NORMAL / PNEUMONIA
        │
        ▼
Grad-CAM (explainability on ConvNeXt features)
# Project Structure
pneumonia_convnext_lstm/
├── dataset/              train/, val/, test/  (NORMAL + PNEUMONIA subfolders)
├── results/              evaluation plots and Grad-CAM figures
├── saved_models/         best_model.pth
├── config.py             hyperparameters and paths
├── prepare_dataset.py    Kaggle data organizer + val split
├── dataset.py            PyTorch Dataset and DataLoaders
├── model.py              ConvNeXtLSTMClassifier
├── train.py              training with AMP, AdamW, early stopping
├── test.py               metrics, ROC/PR curves, confusion matrix
├── gradcam.py            native hook-based Grad-CAM
├── predict.py            single-image inference
├── visualization.py      multi-panel Grad-CAM grid
└── requirements.txt
Quick Start
1. Create and activate a virtual environment
cd "C:\Users\admin\Desktop\SEMESTER 5\DEEPLEARNING\PROJECT\Disease_Detection_CNN"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
2. Install dependencies
cd pneumonia_convnext_lstm
pip install -r requirements.txt
For GPU training, install the CUDA build of PyTorch if needed:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
3. Download the Kaggle dataset
Option A — Kaggle website

Download from Kaggle Chest X-Ray Pneumonia.
Extract the archive. You should get a chest_xray/ folder:
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
Option B — Kaggle CLI

pip install kaggle
# Place kaggle.json in %USERPROFILE%\.kaggle\
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
Expand-Archive chest-xray-pneumonia.zip -DestinationPath .
4. Prepare train / val / test splits
The raw Kaggle dataset has no validation split. Use prepare_dataset.py to create one (15% stratified split from train):

python prepare_dataset.py "C:\path\to\chest_xray"
Options:

Flag	Description
--val-ratio 0.15	Validation fraction (default 15%)
--seed 42	Reproducible split seed
--symlink	Symlink instead of copy (saves disk space)
--no-clear	Keep existing dataset folders
Expected output layout:

dataset/
├── train/   ~4,350 images
├── val/     ~770 images
└── test/    624 images (original Kaggle test)
5. Train the model
python train.py
Training settings (from config.py):

Setting	Value
Batch size	32
Epochs	20 (early stopping patience 5)
Learning rate	1e-4 (AdamW + CosineAnnealingLR)
AMP	Enabled on CUDA
Checkpoint	saved_models/best_model.pth
6. Evaluate on the test set
python test.py
Outputs in results/:

evaluation_report.txt
confusion_matrix.png
roc_curve.png
precision_recall_curve.png
7. Single-image prediction with Grad-CAM
python predict.py dataset/test/PNEUMONIA/person100_bacteria483.jpeg
Saves a side-by-side figure to results/predictions/.

8. Research visualization grid
python visualization.py
Saves results/gradcam_visual_grid.png with 4 NORMAL + 4 PNEUMONIA Grad-CAM examples.

Metrics Reported
Metric	Description
Accuracy	Overall correct classifications
Precision	TP / (TP + FP)
Sensitivity	Recall / True Positive Rate
Specificity	True Negative Rate
F1-Score	Harmonic mean of precision and recall
ROC-AUC	Area under the ROC curve
PNEUMONIA is treated as the positive class (index 1).

Hardware Notes
CUDA GPU — recommended; AMP enabled automatically
Apple Silicon (MPS) — supported; AMP disabled
CPU — works but training is slow; NUM_WORKERS=0 on CPU
Device is selected automatically in config.py.

Reproducibility
Random seed 42 is used across Python, NumPy, and PyTorch. Set via config.SEED.

Citation
If you use this project academically, cite the original dataset:

Mooney, P. (2018). Chest X-Ray Images (Pneumonia) [Dataset]. Kaggle. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

License
This code is for educational and research purposes. The Kaggle dataset has its own license — review it on the dataset page before redistribution.
