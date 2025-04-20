import torch
from fvcore.nn import FlopCountAnalysis
from torchvision.transforms.v2 import Compose, Resize, Normalize, ToImage, ToDtype

# === MODEL IMPORTS ===
from afformer_tiny import AFFormerTiny
from attention_unet import AttentionUNet
from attention_unet_pretrained import AttentionUNet as AttentionUNetPretrained
from bowlnet import BowlNet
from unet import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === TRANSFORM CONFIG ===
TRANSFORM_CONFIG = {
    "afformer-tiny": Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "afformer-tiny-batch32": Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "attention--unet-training": Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "attention--unet-training-pretrained-end-with-conv-scheduler": Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-64batch-512px": Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "BowlNet_64_batch_512px": Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
    "unet-training": Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ]),
}

# === MODEL DEFINITIONS ===
MODELS = {
    "afformer-tiny": AFFormerTiny(in_channels=3, n_classes=19),
    "afformer-tiny-batch32": AFFormerTiny(in_channels=3, n_classes=19),
    "attention--unet-training": AttentionUNet(in_channels=3, n_classes=19),
    "attention--unet-training-pretrained-end-with-conv": AttentionUNetPretrained(in_channels=3, n_classes=19),
    "attention--unet-training-pretrained-end-with-conv-scheduler": AttentionUNetPretrained(in_channels=3, n_classes=19),
    "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-128batch-512px": AttentionUNetPretrained(in_channels=3, n_classes=19),
    "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-64batch-512px": AttentionUNetPretrained(in_channels=3, n_classes=19),
    "BowlNet_64_batch_512px": BowlNet(in_channels=3, n_classes=19),
    "unet-training": UNet(in_channels=3, n_classes=19),
}

def get_input_size_from_transform(transform):
    for t in transform.transforms:
        if isinstance(t, Resize):
            return (1, 3, t.size[0], t.size[1])  # B, C, H, W
    raise ValueError("No Resize found in transform.")

def count_flops(model, input_tensor):
    model.eval()
    flops = FlopCountAnalysis(model, input_tensor)
    return flops.total() / 1e9  # GFLOPs

def main():
    for name, model in MODELS.items():
        model = model.to(DEVICE)
        transform = TRANSFORM_CONFIG.get(name)
        if transform is None:
            print(f"{name}: Skipped — No transform config.")
            continue

        try:
            input_size = get_input_size_from_transform(transform)
            dummy_input = torch.randn(*input_size).to(DEVICE)
            flops = count_flops(model, dummy_input)
            print(f"{name}: {flops:.2f} GFLOPs")
        except Exception as e:
            print(f"{name}: FLOPs calculation failed — {e}")

if __name__ == "__main__":
    main()
