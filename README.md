# Breast Cancer Detection with Histopathological Images

A deep learning project for detecting breast cancer from histopathological images using a custom CNN model named **CancerNet**. This repo provides tools to preprocess the dataset, train the model, log performance metrics, and evaluate predictions.

---

## 🔍 Project Overview

This project performs binary classification (Benign vs. Malignant) of histopathological image patches. Features:
- Dataset preparation and folder splitting
- CNN model (CancerNet) training with augmentations
- Metrics: accuracy, sensitivity, specificity, confusion matrix
- Visualizations: training curves & confusion matrix heatmap

---

## 📁 Dataset

Dataset used: [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/data)

### Setup:
1. Download and unzip the dataset.
2. Place it in the directory configured in the root folder.

---

## 🗂️ Project Structure

```
├── cancernet/                  # Model architecture and config
├── src/
│   ├── build_dataset.py        # Script to split dataset
│   ├── model.py                # Train, evaluate, and save the model
│   ├── training_details.txt    # Logs for each epoch
│   ├── plot.png                # Accuracy/Loss curves
│   ├── confusion_matrix.png    # Confusion matrix heatmap
│   ├── model_last_epoch_*.keras/.h5  # Saved models
├── requirements.txt            # Dependencies
└── README.md
```

---

## 🚀 Getting Started

### 🔗 Clone the Repository

```bash
git clone https://github.com/08anask/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
```

### 🛠️ Set Up the Environment

```bash
pip install -r requirements.txt
```

---

## ⚙️ How to Run

### Step 1️⃣: Prepare Dataset

```bash
python src/build_dataset.py
```

This will split the dataset into `training/`, `validation/`, and `testing/` folders as per config.

### Step 2️⃣: Train the Model

```bash
python src/model.py
```

This will:
- Train CancerNet with augmentation and early stopping
- Save logs, plots, model files (`.keras` & `.h5`)
- Evaluate with accuracy, sensitivity, specificity

---

## 📊 Model Performance

Metrics computed:
- **Accuracy**
- **Sensitivity** (Recall of positive class)
- **Specificity** (Recall of negative class)

📈 Outputs:
- `plot.png`: Loss & accuracy curves
- `confusion_matrix.png`: Confusion matrix visualization

📜 Full training log excerpt:

```text
Epoch 1/50 - train_loss: 0.5890 - val_loss: 0.6177 - train_acc: 0.8224 - val_acc: 0.7490
Epoch 2/50 - train_loss: 0.5441 - val_loss: 0.4981 - train_acc: 0.8359 - val_acc: 0.8103
...
Epoch 50/50 - train_loss: 0.4491 - val_loss: 0.3931 - train_acc: 0.8661 - val_acc: 0.8435
```

For complete logs, check **[training_details.txt](src/training_details.txt)**

---

## 🧠 CancerNet Architecture

Located in `cancernet/cancernet.py`:
- Input: 48x48 RGB images
- Output: 2 classes (Benign, Malignant)
- Framework: TensorFlow/Keras

---

## 📦 Dependencies

Listed in `requirements.txt`:
- TensorFlow / Keras
- NumPy / OpenCV / PIL
- scikit-learn / imutils
- matplotlib / seaborn

Install with:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes
- Uses **Adagrad** optimizer (can be switched to Adam)
- Uses `EarlyStopping` and custom epoch logger
- Model saved with timestamp for reproducibility

---

## 📝 License

Licensed under the MIT License.  
See the [LICENSE](LICENSE) file for more information.

---

## 💬 Contact

Open an issue or pull request to contribute or raise questions.

---

Created with ❤️ to support research and awareness in breast cancer detection.