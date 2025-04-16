import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms.v2 import functional as F
from segment_anything import sam_model_registry

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels
def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image

class CityscapesTransform:
    def __init__(self):
        self.size = (1024, 1024)

    def __call__(self, sample):
        image, target = sample["image"], sample["target"]
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return {"image": image, "target": target}

# SAM loading
def load_sam_model(checkpoint_path: str, device: torch.device):
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path).to(device)
    return sam.image_encoder

# Decoder
class SAMSegmentationDecoder(nn.Module):
    def __init__(self, encoder_out_dim=1280, num_classes=19):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_out_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)

class SAMSegmentationModel(nn.Module):
    def __init__(self, encoder, num_classes=19):
        super().__init__()
        self.encoder = encoder
        self.decoder = SAMSegmentationDecoder(encoder_out_dim=1280, num_classes=num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        out = self.decoder(feats)
        return out

def get_args_parser():
    parser = ArgumentParser("Training SAM-based segmentation model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="sam-vit-h-transfer")
    parser.add_argument("--sam-checkpoint", type=str, required=True, help="Path to sam_vit_h_4b8939.pth")
    return parser

def main(args):
    wandb.init(project="5lsm0-cityscapes-segmentation", name=args.experiment_id, config=vars(args))

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = CityscapesTransform()

    train_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=transform)
    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=transform)

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    encoder = load_sam_model(args.sam_checkpoint, device)
    model = SAMSegmentationModel(encoder=encoder, num_classes=19).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:02}/{args.epochs:02}")
        model.train()

        for i, (images, labels) in enumerate(train_loader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device).long().squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({"train_loss": loss.item(), "epoch": epoch + 1}, step=epoch * len(train_loader) + i)

        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(val_loader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device).long().squeeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                if i == 0:
                    predictions = outputs.softmax(1).argmax(1).unsqueeze(1)
                    labels = labels.unsqueeze(1)
                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)
                    wandb.log({
                        "predictions": [wandb.Image(make_grid(predictions.cpu(), nrow=4))],
                        "labels": [wandb.Image(make_grid(labels.cpu(), nrow=4))]
                    }, step=(epoch + 1) * len(train_loader) - 1)

            valid_loss = sum(losses) / len(losses)
            wandb.log({"valid_loss": valid_loss}, step=(epoch + 1) * len(train_loader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(output_dir, f"best_model-epoch={epoch:02}-val_loss={valid_loss:.4f}.pth"))

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr}, step=(epoch + 1) * len(train_loader) - 1)

    print("Training complete!")
    torch.save(model.state_dict(), os.path.join(output_dir, f"final_model-epoch={epoch:02}-val_loss={valid_loss:.4f}.pth"))
    wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
