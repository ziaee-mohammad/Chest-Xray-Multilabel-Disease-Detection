# 🩻 Chest X‑ray Multilabel Disease Detection

A carefully engineered **multilabel** Computer Vision project for detecting multiple thoracic findings from **chest X‑ray** images.  
The pipeline includes **patient‑wise splitting**, robust preprocessing/augmentation, **CNN backbones** (DenseNet/ResNet via `timm`), **class‑imbalance handling**, **per‑class threshold tuning**, and clinical‑style evaluation (**AUROC, mAP, sensitivity @ specificity**).  
Interpretability is provided with **Grad‑CAM** heatmaps.

> ⚠️ Medical disclaimer: this repository is for **research & education** only and must **not** be used for clinical decision‑making.

---

## 🧠 Problem
Given a CXR image, predict **multiple labels simultaneously** (e.g., *Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax*).
This is a **multi‑label, highly imbalanced** classification task; metrics must be **per‑label** and **macro/micro** averaged.

---

## 🗂️ Dataset
- Source: public chest X‑ray datasets (e.g., **ChestX‑ray14 / CheXpert / VinDr‑CXR**). Replace links/paths with the dataset you used.  
- Input format: DICOM or PNG/JPEG; images are center‑cropped/resized.  
- **Patient‑wise** train/val/test split (no patient leakage).  
- Labels: 14 binary findings (1/0), possibly multi‑hot per image.  
- Recommended image size: **224–320** px (trade‑off accuracy vs speed).

> If using CheXpert's uncertainty labels (U), choose a policy (e.g., **U→0**, **U→1**, or **ignore** during loss). Document it here.

---

## 🧰 Pipeline Overview
1. **Load & split** by *patient ID*  
2. **Preprocessing**: convert to RGB, normalize to ImageNet mean/std  
3. **Augmentation** (train): RandomResizedCrop/Resize, HorizontalFlip (careful with laterality), Brightness/Contrast, CLAHE (optional)  
4. **Backbone**: `timm` models (e.g., **DenseNet121**, **ResNet50**, **ConvNeXt‑Tiny**), global pooling, **sigmoid** outputs (C = #classes)  
5. **Loss**: `BCEWithLogitsLoss` (optionally **Focal** / **Class‑Balanced** weights)  
6. **Optimization**: AdamW + cosine schedule / ReduceLROnPlateau  
7. **Thresholds**: per‑class, tuned on **validation** to maximize F1 or Youden‑J  
8. **Evaluation**: AUROC (per class + macro/micro), **mAP**, **Sensitivity@Spec** (e.g., Sen@Spec=0.80)  
9. **Explainability**: **Grad‑CAM** for positive findings

---

## 🧪 Classes (example 14‑label set)
```
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis,
Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax, No_Finding
```
> Adjust to your dataset’s taxonomy. If `No_Finding` exists, document how you treat conflicts with other labels.

---

## 📈 Metrics & Reporting
- **AUROC (per‑class)**, **macro AUROC**, **micro AUROC**  
- **Average Precision (mAP)** (macro)  
- **Sensitivity / Specificity** at tuned thresholds (per class)  
- Optional: **Calibration** (Platt/Isotonic) per label

**Results (replace with your numbers):**
| Class | AUROC | AP | Thr | Sen@Spec(0.8) |
|------|------:|---:|----:|---------------:|
| Atelectasis | 0.86 | 0.47 | 0.41 | 0.72 |
| Cardiomegaly | 0.92 | 0.63 | 0.38 | 0.81 |
| … | … | … | … | … |
| **Macro** | **0.88** | **0.52** | — | — |

---

## 🧩 Repository Structure (suggested)
```
Chest-Xray-Multilabel-Disease-Detection/
├─ notebooks/
│  ├─ 01_explore.ipynb
│  ├─ 02_train.ipynb
│  ├─ 03_eval_cam.ipynb
├─ src/
│  ├─ data.py          # Dataset/Dataloader (patient-wise split)
│  ├─ transforms.py    # Augmentations (Albumentations / torchvision)
│  ├─ models.py        # timm backbones → multi-label head
│  ├─ losses.py        # BCEWithLogits, Focal, class-balanced
│  ├─ train.py         # train loop, AMP, early-stop
│  ├─ infer.py         # predict single image / folder
│  ├─ eval.py          # AUROC/AP, threshold sweep, Sen@Spec
│  ├─ cam.py           # Grad-CAM visualizations
│  └─ utils.py
├─ configs/            # YAMLs for experiments
├─ reports/figures/    # Curves, CAMs, sample predictions
├─ data/               # (gitignored) metadata, splits
├─ models/             # (gitignored) checkpoints
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## ⚙️ Installation
```bash
git clone https://github.com/ziaee-mohammad/Chest-Xray-Multilabel-Disease-Detection.git
cd Chest-Xray-Multilabel-Disease-Detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt (example)**
```
torch
torchvision
timm
albumentations
opencv-python
pandas
numpy
scikit-learn
matplotlib
seaborn
pytorch-grad-cam
pyyaml
```

---

## 🚀 Usage

### 1) Train
```bash
python -m src.train   --data_dir data/chexpert   --train_csv splits/train.csv   --val_csv   splits/val.csv   --classes   configs/classes.txt   --model     densenet121   --img_size  320   --batch     32   --loss      bce   --epochs    20   --amp
```

### 2) Evaluate & Threshold Tuning
```bash
python -m src.eval   --ckpt models/densenet121_best.pt   --val_csv splits/val.csv   --metrics auroc ap   --tune_threshold f1  # or youden
```

### 3) Inference (single image / folder)
```bash
python -m src.infer   --ckpt models/densenet121_best.pt   --source path/to/image_or_dir   --out   outputs/preds.csv
```

### 4) Grad‑CAM
```bash
python -m src.cam   --ckpt models/densenet121_best.pt   --image path/to/cxr.png   --target Effusion   --save  reports/figures/cam_effusion.png
```

---

## 🔬 Implementation Notes
- **Patient‑wise split** is mandatory to avoid leakage.  
- Normalize to ImageNet mean/std; keep PNG/JPEG in 8‑bit.  
- Horizontal flip may invert laterality—consider disabling when labels depend on left/right.  
- Track **class prevalence**; log‐scale loss weights or **Focal** loss may help rare classes.  
- **Per‑class thresholds** generally outperform a single global threshold.  
- Report **macro/micro** scores; per‑class performance is essential for clinical auditing.  
- (Optional) Temperature scaling / isotonic **calibration** per label.

---

## 📜 Ethics & Privacy
- Remove PHI; do not publish patient identifiers.  
- Follow dataset licenses and cite properly.  
- This code is not a medical device; results are **not** clinical grade.

---

## 👤 Author
**Mohammad Ziaee** — Computer Engineer | AI & Data Science  
📧 moha2012zia@gmail.com  
🔗 https://github.com/ziaee-mohammad
👉 Instagram: [@ziaee_mohammad](https://www.instagram.com/ziaee_mohammad/)

---

## 🏷 Tags
```
data-science
machine-learning
deep-learning
computer-vision
medical-imaging
multi-label-classification
chest-xray
chexpert
pytorch
grad-cam
timm
```
