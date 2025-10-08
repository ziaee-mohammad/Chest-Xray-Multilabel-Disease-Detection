import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Paths
CSV_PATH = "/kaggle/input/data/Data_Entry_2017.csv"
DATA_DIR = "/kaggle/input/data/"

# Diseases to classify
TARGET_DISEASES = ["Atelectasis", "Cardiomegaly", "Effusion",
                   "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]


class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_root = image_root
        self.transform = transform
        self.data["TargetVector"] = self.data["Finding Labels"].apply(self.encode_labels)

    def encode_labels(self, label_str):
        labels = label_str.split('|')
        return [1 if disease in labels else 0 for disease in TARGET_DISEASES]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row["Image Index"]
        img_path = self.find_image_path(img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.FloatTensor(row["TargetVector"])
        return image, label

    def find_image_path(self, img_name):
        for folder in os.listdir(self.image_root):
            candidate = os.path.join(self.image_root, folder, "images", img_name)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"{img_name} not found.")

# LSE Pooling
class LSEPooling(nn.Module):
    def __init__(self, r=10, eps=1e-6):
        super(LSEPooling, self).__init__()
        self.r = r
        self.eps = eps

    def forward(self, x):
        r = self.r
        x_max = torch.amax(x.view(x.size(0), x.size(1), -1), dim=2, keepdim=True)
        x_exp = torch.exp(r * (x.view(x.size(0), x.size(1), -1) - x_max))
        x_lse = x_max + (1.0 / r) * torch.log(self.eps + x_exp.mean(dim=2, keepdim=True))
        return x_lse.squeeze(-1)

# Model
class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=8, r=10):
        super(ChestXrayModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.transition = nn.Conv2d(2048, 1024, kernel_size=1)
        self.pooling = LSEPooling(r)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.transition(x)
        x = self.pooling(x)
        x = self.classifier(x)
        return torch.sigmoid(x)

# Weighted BCE Loss
def weighted_bce_loss(output, target):
    eps = 1e-8
    pos_weight = (target == 0).float().sum() / ((target == 1).float().sum() + eps)
    loss = - (pos_weight * target * torch.log(output + eps) +
              (1 - target) * torch.log(1 - output + eps))
    return loss.mean()

# Training
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    dataset = NIHChestXrayDataset(CSV_PATH, DATA_DIR, transform)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    model = ChestXrayModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):  # Train for 5 epochs
        model.train()
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = weighted_bce_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}], Step [{i}], Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch+1}] completed with avg loss: {total_loss / len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "/kaggle/working/chestxray_model_resnet50.pth")
    print("Model saved as chestxray_model_resnet50.pth")

# Run
if __name__ == "__main__":
    train_model()