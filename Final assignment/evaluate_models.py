import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
# from fvcore.nn import FlopCountAnalysis, parameter_count
from transforms_config import TRANSFORM_CONFIG
from torchinfo import summary
import numpy as np
from PIL import Image

# Import models
from afformer_tiny import AFFormerTiny  # Import your other models as needed
from attention_unet import AttentionUNet
from attention_unet_pretrained import AttentionUNet as AttentionUNetPretrained 
from bowlnet import BowlNet
from unet import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CITYSCAPES_ID_TO_TRAINID = {
     0: 255,
     1: 255,
     2: 255,
     3: 255,
     4: 255,
     5: 255,
     6: 255,
     7: 0,    # road
     8: 1,    # sidewalk
     9: 255,
    10: 255,
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,
    15: 255,
    16: 255,
    17: 5,    # pole
    18: 255,
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,
    30: 255,
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
    -1: 255,
}

def convert_to_train_ids(label):
    label = np.array(label)
    label_copy = 255 * np.ones_like(label, dtype=np.uint8)
    for k, v in CITYSCAPES_ID_TO_TRAINID.items():
        label_copy[label == k] = v
    return Image.fromarray(label_copy)

class CityscapesTrainIDWrapper(Cityscapes):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        label = convert_to_train_ids(label)
        return image, label

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

# def count_flops_params(model, sample_input):
#     flops = FlopCountAnalysis(model, sample_input)
#     params = parameter_count(model)
#     return flops.total() / 1e9, params[""] / 1e6  # GFLOPs and MParams

def get_param_count(model, input_size):
    try:
        model.eval()
        info = summary(model, input_size=input_size, verbose=0)
        return info.total_params / 1e6  # Return in MParams
    except Exception as e:
        print(f"Could not summarize model: {e}")
        return -1

def get_model_by_folder(folder_name):
    if "afformer-tiny" in folder_name:
        print("------------afformer_1---------------")
        return AFFormerTiny(in_channels=3, n_classes=19)
    elif "afformer-tiny-batch32" in folder_name:
        print("---------------afformer_2(batch32)------------")
        return AFFormerTiny(in_channels=3, n_classes=19)
    # elif "attention--unet-training" in folder_name:
    #     return AttentionUNet(in_channels=3, n_classes=19)
    elif "attention--unet-training-pretrained-end-with-conv" in folder_name:
        print(1)
        return AttentionUNetPretrained(in_channels=3, n_classes=19)
    elif "attention--unet-training-pretrained-end-with-conv-scheduler" in folder_name:
        print("2_sheduler")
        return AttentionUNetPretrained(in_channels=3, n_classes=19)
    elif "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-128batch-512px" in folder_name:
        print(3)
        return AttentionUNetPretrained(in_channels=3, n_classes=19)
    elif "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-64batch-512px" in folder_name:
        print(4)
        return AttentionUNetPretrained(in_channels=3, n_classes=19)
    elif "BowlNet_64_batch_512px" in folder_name:
        print("bowl")
        return BowlNet(in_channels=3, n_classes=19)
    elif "unet-training" in folder_name:
        print("unet")
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

        val_set = CityscapesTrainIDWrapper(
            "./data/cityscapes", 
            split="val", 
            mode="fine", 
            target_type="semantic", 
            transforms=transform
        )
        val_set = wrap_dataset_for_transforms_v2(val_set)
        val_loader = DataLoader(val_set, batch_size=4, shuffle=False)
        model = get_model_by_folder(folder).to(DEVICE)
        model.load_state_dict(torch.load(load_best_model_path(folder_path), map_location=DEVICE))

        metrics = evaluate_model(model, val_loader)
        sample_input = next(iter(val_loader))[0].to(DEVICE)
        # flops, params = count_flops_params(model, sample_input)
        param_count = get_param_count(model, input_size=sample_input.shape)

        results.append({
            "folder": folder,
            **metrics,
            # "FLOPs (G)": round(flops, 2),
            # "Params (M)": round(params, 2)
            "Params (M)": round(param_count, 2) if param_count != -1 else "N/A"
        })
        print(results)

    print("\n=== Evaluation Summary ===")
    for res in results:
        print(res)

    if skipped_folders:
        print("\n Skipped Folders:")
        for f in skipped_folders:
            print(f"- {f}")

if __name__ == "__main__":
    main()
