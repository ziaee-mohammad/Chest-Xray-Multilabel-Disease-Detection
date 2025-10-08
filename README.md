
# 🧠 Chest X-ray Analysis using Deep Learning

A third year Artificial Intelligence project developed at Bahria University Islamabad that leverages deep learning (ResNet-50) for multi-label classification of thoracic diseases from chest X-ray images. The system includes Grad-CAM visualizations to aid radiologists in understanding model decisions.

---

## 📌 Project Overview

Radiologists often face challenges interpreting complex X-ray images. This AI-powered diagnostic tool assists in detecting various thoracic diseases like Pneumonia, Edema, Cardiomegaly, Infiltration etc using deep learning techniques. It aims to enhance diagnostic accuracy and reduce human error in medical imaging.

---

## 📷 Dataset

**NIH ChestX-ray14**  
- 112,120 frontal-view X-ray images  
- 30,805 unique patients  
- 14 disease labels (multi-label)  
- Image Size: 1024x1024

**Key Pathologies Covered:**
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Fibrosis
- Hernia
- Pleural Thickening
- Emphysema

---

## 🏗️ Model Architecture

- Base Model: **ResNet-50** (Pretrained on ImageNet)
- Final Layers:
  - `Conv2D(2048 → 1024)`
  - **LSE Pooling** (`r=10`)
  - `FC(1024 → 8)` with **Sigmoid activation**
- Loss Function: **Weighted Binary Cross-Entropy**
- Evaluation Metrics:
  - Accuracy
  - Precision, Recall, F1-Score (per class)
  - ROC-AUC
  - Confusion Matrix

---

## 🧪 Training Details

| Parameter      | Value           |
|----------------|------------------|
| Optimizer      | Adam             |
| Learning Rate  | 1e-4             |
| Batch Size     | 16               |
| Epochs         | 5                |
| Device         | Kaggle T4 GPU    |
| Input Size     | 512×512 RGB      |

Model checkpoint: `chestxray_model_resnet50.pth`

---

## 🧠 Model Performance

| Model           | AUC    | Accuracy |
|------------------|--------|----------|
| **ResNet-50**     | 0.72   | 0.78     |
| EfficientNet B1  | 0.75   | 0.53     |
| DenseNet         | 0.32   | 0.33     |

Grad-CAM visualizations are provided for visual explainability, highlighting affected lung regions.

---

## 🖥️ Frontend Features

- Upload Chest X-ray image
- Get multi-label disease predictions
- View Grad-CAM heatmaps for interpretability

---

## 🎯 Objectives

- Detect thoracic diseases from chest X-rays using deep learning
- Provide visual interpretability using Grad-CAM
- Assist radiologists in faster, more accurate diagnosis

---

## 👤 Target Users

- Radiologists
- Medical researchers
- Healthcare institutions (especially resource-limited settings)

---

## 🛠️ Installation

1. Clone the repository:
   ```
   git clone https://github.com/awab-sial/Chest-X-ray-Analysis-using-Deep-Learning
   cd Chest-X-ray-Analysis-using-Deep-Learning

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (optional if using pretrained):

   ```bash
   python train.py
   ```

4. Launch frontend:

   ```bash
   python main.py
   ```




## 📁 Repository Structure

```
├── 📁 Model_Training       # Contains training and evaluation logic
│   ├── train.py            # Script to train the model
│   └── test.py             # Script to test the model on a test set
├── 📁 static               # Stores static files like images, CSS, or Grad-CAM outputs
├── 📁 templates            # HTML templates for web interface
├── main.py                 # Entry point to run training, testing, or predictions
├── predictions.py          # Contains logic for making predictions on new data
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and usage instructions
├── Report.pdf              # Final project report/documentation
└── .gitignore              # Specifies files and folders to ignore in Git

```

## LinkedIn Demo

This project has been showcased on my LinkedIn profile.  
You can view detailed updates, project insights, and discussions there.

Check it out here:  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Demo-blue?logo=linkedin&style=for-the-badge)](https://tinyurl.com/392wh5z9)


---

## 👥 Contributors

* **Muhammad Awab Sial** – [@awab-sial](https://github.com/awab-sial)
* **Syed Amber Ali Shah**  – [@Amber-Ali-Shah](https://github.com/Amber-Ali-Shah)

Supervised by: 
* **Dr. Arshad Farhad**
* **Ms. Ayeza Ghazanfar**
* **Ms. Mehroz Sadiq**

---

## 📄 License

This project is for academic purposes and is not intended for real clinical use.

---

## 💬 Feedback

If you find this useful or have suggestions, feel free to open an issue or a pull request.

