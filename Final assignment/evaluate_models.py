import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from fvcore.nn import FlopCountAnalysis, parameter_count
from transforms_config import TRANSFORM_CONFIG

# Import models
from afformer_tiny import AFFormerTiny  # Import your other models as needed
from attention_unet import AttentionUNet
from attention_unet_pretrained import AttentionUNet as AttentionUNetPretrained 
from bowlnet import BowlNet
from unet import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_best_model_path(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".pth")]
    best = sorted(files, key=lambda f: float(f.split("val_loss=")[-1].replace(".pth", "")))[0]
    return os.path.join(folder, best)

def evaluate_model(model, dataloader):
    model.eval()
    iou_metric = MulticlassJaccardIndex(num_classes=19, ignore_index=255).to(DEVICE)
    acc_metric = MulticlassAccuracy(num_classes=19, ignore_index=255).to(DEVICE)

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            labels = labels.long().squeeze(1)
            outputs = model(imgs).argmax(1)
            iou_metric.update(outputs, labels)
            acc_metric.update(outputs, labels)

    return {
        "IoU": iou_metric.compute().item(),
        "Accuracy": acc_metric.compute().item()
    }

def count_flops_params(model, sample_input):
    flops = FlopCountAnalysis(model, sample_input)
    params = parameter_count(model)
    return flops.total() / 1e9, params[""] / 1e6  # GFLOPs and MParams

def get_model_by_folder(folder_name):
    if "afformer-tiny" in folder_name:
        return AFFormerTiny(in_channels=3, n_classes=19)
    elif "afformer-tiny-batch32" in folder_name:
        return AFFormerTiny(in_channels=3, n_classes=19)
    elif "attention--unet-training" in folder_name:
        return AttentionUNet(in_channels=3, n_classes=19)
    elif "attention--unet-training-pretrained-end-with-conv" in folder_name:
        return AttentionUNetPretrained(in_channels=3, n_classes=19)
    elif "attention--unet-training-pretrained-end-with-conv-scheduler" in folder_name:
        return AttentionUNetPretrained(in_channels=3, n_classes=19)
    elif "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-128batch-512px" in folder_name:
        return AttentionUNetPretrained(in_channels=3, n_classes=19)
    elif "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-64batch-512px" in folder_name:
        return AttentionUNetPretrained(in_channels=3, n_classes=19)
    elif "BowlNet_64_batch_512px" in folder_name:
        return BowlNet(in_channels=3, n_classes=19)
    elif "unet-training" in folder_name:
        return UNet(in_channels=3, n_classes=19)
    # add more models here (be careful if they are missing!)
    raise ValueError(f"Unknown model for folder: {folder_name}")

def main():
    base_dir = "checkpoints"
    results = []
    skipped_folders = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Skip if no transform
        transform = TRANSFORM_CONFIG.get(folder, None)
        if transform is None:
            print(f"Skipping {folder}: No transform config found.")
            skipped_folders.append(folder)
            continue

        # Skip if unknown model
        model = get_model_by_folder(folder)
        if model is None:
            print(f"Skipping {folder}: No model matched.")
            skipped_folders.append(folder)
            continue

        val_set = Cityscapes("./data/cityscapes", split="val", mode="fine", target_type="semantic", transforms=transform)
        val_set = wrap_dataset_for_transforms_v2(val_set)
        val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

        model = get_model_by_folder(folder).to(DEVICE)
        model.load_state_dict(torch.load(load_best_model_path(folder_path), map_location=DEVICE))

        metrics = evaluate_model(model, val_loader)
        sample_input = next(iter(val_loader))[0].to(DEVICE)
        flops, params = count_flops_params(model, sample_input)

        results.append({
            "folder": folder,
            **metrics,
            "FLOPs (G)": round(flops, 2),
            "Params (M)": round(params, 2)
        })

    print("\n=== Evaluation Summary ===")
    for res in results:
        print(res)

if __name__ == "__main__":
    main()
