🫁 Chest X-ray Multilabel Disease Detection

AI system for multi-label classification of chest X-ray images using deep learning and Grad-CAM visualization to assist in disease interpretation.

🔍 Overview

This project detects multiple thoracic diseases (such as Pneumonia, Edema, Cardiomegaly, and Infiltration) from chest X-rays using a ResNet-50 model.
It aims to improve diagnostic support and provide explainable visual outputs for radiologists.

🧠 Dataset

NIH ChestX-ray14

112,120 images, 30,000+ patients

14 disease categories (multi-label)

Size: 1024×1024 pixels


⚙ Model Details

Backbone: ResNet-50 (ImageNet pretrained)

Pooling: LSE pooling

Loss: Weighted Binary Cross-Entropy

Evaluation: Accuracy, ROC-AUC, Precision/Recall/F1

Explainability: Grad-CAM heatmaps


🧪 Training Setup

Optimizer: Adam (lr=1e-4)

Epochs: 5

Batch size: 16

GPU: Kaggle T4


📊 Performance

ResNet-50 achieved AUC ≈ 0.72 and accuracy ≈ 0.78.
Grad-CAM visualizations show affected lung areas for interpretability.

🖥 Features

Upload X-ray image

Multi-label disease prediction

Grad-CAM visualization

Web UI for interaction


⚠ Note

This model is for research and educational use only and not intended for clinical diagnosis.

👨🏻‍💻 Author

Mohammad Ziaee
Moha2012zia@gmail.com



