import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

# Class labels
TARGET_DISEASES = ["Atelectasis", "Cardiomegaly", "Effusion",
                   "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

# LSE Pooling Layer
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
        self.backbone = models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.transition = nn.Conv2d(2048, 1024, kernel_size=1)
        self.pooling = LSEPooling(r)
        self.classifier = nn.Linear(1024, num_classes)
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        self.feature_maps = self.features(x)
        x = self.transition(self.feature_maps)
        x.register_hook(self.activations_hook)
        pooled = self.pooling(x)
        out = self.classifier(pooled)
        return torch.sigmoid(out)

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.feature_maps

# Load model
def load_model(model_path):
    model = ChestXrayModel()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()
    return model
    
# Generate Grad-CAM
def generate_gradcam(model, image_tensor, class_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.enable_grad():
        output = model(image_tensor)
        score = output[0, class_idx]
        model.zero_grad()
        score.backward(retain_graph=True)

        gradients = model.get_activations_gradient()[0]
        activations = model.get_activations()[0]

        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], device=device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max()
        cam = cam.detach().cpu().numpy()
        return cam

# Grad-CAM visualization
def visualize_gradcams(model, image_path, threshold=0.5, top_k=3):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    pil_image = Image.open(image_path).convert("RGB").resize((512, 512))
    image_tensor = transform(pil_image).unsqueeze(0)
    image_tensor.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.enable_grad():
        output = model(image_tensor)
        probs = output[0].detach().cpu().numpy()

    # Get all indices where probability > threshold
    above_thresh_indices = [(i, p) for i, p in enumerate(probs) if p > threshold]
    
    # Sort by probability in descending order and take top_k
    top_predictions = sorted(above_thresh_indices, key=lambda x: x[1], reverse=True)[:top_k]

    if not top_predictions:
        print("No predictions above threshold.")
        return

    for class_idx, prob in top_predictions:
        cam = generate_gradcam(model, image_tensor.clone(), class_idx)

        # Convert CAM to color
        cam_resized = np.uint8(255 * cam)
        heatmap = Image.fromarray(cam_resized).resize((512, 512), resample=Image.BILINEAR)
        heatmap = np.array(heatmap)
        heatmap = plt.cm.jet(heatmap)[:, :, :3]  # RGB heatmap only

        # Convert PIL image to array
        base_img = np.array(pil_image) / 255.0

        # Overlay heatmap on image
        overlay = 0.5 * heatmap + 0.5 * base_img
        overlay = np.clip(overlay, 0, 1)

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(f"Grad-CAM: {TARGET_DISEASES[class_idx]} ({prob:.2f})")
        plt.axis("off")
        plt.show()


# Example usage
if __name__ == "__main__":
    image_path = "Replace With Test image"  # Replace with your test image
    model_path = "./chestxray_model_resnet50.pth"

    model = load_model(model_path)
    visualize_gradcams(model, image_path)