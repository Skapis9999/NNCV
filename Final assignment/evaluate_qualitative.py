import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from torchvision import transforms

# Import your model classes
from afformer_tiny import AFFormerTiny
from attention_unet_pretrained import AttentionUNet as AttentionUNetPretrained
from bowlnet import BowlNet
from unet import UNet
from transforms_config import TRANSFORM_CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_best_model_path(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".pth")]
    best = sorted(files, key=lambda f: float(f.split("val_loss=")[-1].replace(".pth", "")))[0]
    return os.path.join(folder, best)

def get_model_by_folder(folder_name):
    if "afformer-tiny" in folder_name:
        return AFFormerTiny(in_channels=3, n_classes=19)
    elif "attention--unet-training-pretrained" in folder_name:
        return AttentionUNetPretrained(in_channels=3, n_classes=19)
    elif "BowlNet" in folder_name:
        return BowlNet(in_channels=3, n_classes=19)
    elif "unet-training" in folder_name:
        return UNet(in_channels=3, n_classes=19)
    raise ValueError(f"Unknown model for folder: {folder_name}")

class InferenceImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        transformed = self.transform(image) if self.transform else self.to_tensor(image)        
        original_tensor = self.to_tensor(image) 
        return transformed, image_path, original_tensor

def visualize_model_predictions(models, dataloader, device=DEVICE, num_images=3):
    # Clean display names for models
    CLEAN_MODEL_NAMES = {
        "afformer-tiny": "AFFormer Tiny",
        "attention--unet-training-pretrained-end-with-conv-scheduler": "Attn UNet",
        "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-64batch-512px": "Attn UNet (Finetuned)",
        "BowlNet_64_batch_512px": "BowlNet",
        "unet-training": "UNet",
    }

    model_outputs = {name: [] for name in models}
    original_images = []
    image_names = []

    # Collect predictions
    for inputs, paths, originals in dataloader:
        inputs = inputs.to(device)
        original_images.extend(originals)
        image_names.extend(paths)

        for model_name, model in models.items():
            model.eval()
            with torch.no_grad():
                preds = model(inputs)
                preds = preds.argmax(1).cpu()
                model_outputs[model_name].extend(preds)

    # Prepare subplot grid
    num_models = len(models)
    total_cols = 1 + num_models  # 1 for input + each model
    total_rows = min(num_images, len(original_images))

    fig, axs = plt.subplots(total_rows, total_cols, figsize=(4 * total_cols, 4 * total_rows))

    # Handle single-row case
    if total_rows == 1:
        axs = [axs]

    # Add column titles with clean model names
    model_keys_sorted = sorted(model_outputs.keys())
    column_titles = ["Input"] + [CLEAN_MODEL_NAMES.get(k, k) for k in model_keys_sorted]
    for col in range(total_cols):
        axs[0][col].set_title(column_titles[col], fontsize=14)

    # Fill in images
    for row in range(total_rows):
        # Input image
        img = original_images[row]
        axs[row][0].imshow(TF.to_pil_image(img))
        axs[row][0].axis("off")

        # Predictions
        for col, model_name in enumerate(model_keys_sorted, start=1):
            pred = model_outputs[model_name][row]
            axs[row][col].imshow(pred)  # Assuming pred is already color-encoded or label-based
            axs[row][col].axis("off")

    plt.tight_layout()
    plt.show()




def main():
    base_dir = "checkpoints"
    selected_folders = [
        "afformer-tiny",
        "attention--unet-training-pretrained-end-with-conv-scheduler",
        "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-64batch-512px",
        "BowlNet_64_batch_512px",
        "unet-training"
    ]

    models = {}
    for folder in selected_folders:
        folder_path = os.path.join(base_dir, folder)
        transform = TRANSFORM_CONFIG.get(folder, None)
        if transform is None:
            print(f"Skipping {folder} due to missing transform.")
            continue

        model = get_model_by_folder(folder).to(DEVICE)
        model.load_state_dict(torch.load(load_best_model_path(folder_path), map_location=DEVICE))
        model.eval()
        models[folder] = model

    transform = TRANSFORM_CONFIG[selected_folders[0]]  # Use transform from first model
    inference_set = InferenceImageDataset("./data/cityscapes", transform=transform)
    inference_loader = DataLoader(inference_set, batch_size=1, shuffle=False)

    visualize_model_predictions(models, inference_loader, num_images=len(inference_set))

if __name__ == "__main__":
    main()
