import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype
import wandb
from argparse import ArgumentParser
from segment_anything import sam_model_registry
import time

# Lightweight segmentation head
class LightSegHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        return self.decode(x)


def get_args_parser():
    parser = ArgumentParser("SAM to Lightweight Transfer Learning")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--experiment-id", type=str, default="sam-light-transfer")
    parser.add_argument("--lr", type=float, default=0.0001)
    return parser


def fine_tune_light_model(args):
    wandb.init(project="sam-cityscapes", name=args.experiment_id, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained SAM model https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    sam.eval()
    for param in sam.parameters():
        param.requires_grad = False

    # Lightweight segmentation head
    light_head = LightSegHead(in_channels=256, num_classes=20).to(device)  # Adjust channels if needed

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(light_head.parameters(), lr=args.lr)

    transform = Compose([
        ToImage(),
        Resize((1024, 1024)),  # Must match what SAM expects
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ])

    train_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=transform)
    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=transform)

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_valid_loss = float("inf")
    model_save_path = os.path.join("checkpoints", f"light_head_{args.experiment_id}.pth")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        sam.eval()
        light_head.train()
        train_loss = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                features = sam.image_encoder(images)  # shape: (B, C, H, W)

            outputs = light_head(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            wandb.log({"train_loss": loss.item(), "epoch": epoch + 1}, step=epoch * len(train_loader) + i)

        # Validation
        light_head.eval()
        valid_loss = 0
        total_infer_time = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                features = sam.image_encoder(images)
                start = time.time()
                outputs = light_head(features)
                end = time.time()
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                total_infer_time += (end - start)

        avg_valid_loss = valid_loss / len(valid_loader)
        avg_infer_time = total_infer_time / len(valid_loader.dataset)
        model_size_mb = sum(p.numel() for p in light_head.parameters()) * 4 / (1024 ** 2)

        wandb.log({
            "valid_loss": avg_valid_loss,
            "avg_inference_time": avg_infer_time,
            "model_size_MB": model_size_mb,
            "epoch": epoch + 1
        })

        print(f"Epoch {epoch+1}: Train Loss={train_loss / len(train_loader):.4f}, Valid Loss={avg_valid_loss:.4f}")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(light_head.state_dict(), model_save_path)
            print(f"Best model saved at {model_save_path}")

    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    fine_tune_light_model(args)
