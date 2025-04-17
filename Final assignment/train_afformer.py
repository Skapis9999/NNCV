import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
)
from torch.optim.lr_scheduler import StepLR

from afformer_tiny import AFFormerTiny  # Importing the AFFormer-tiny model

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

def get_args_parser():
    parser = ArgumentParser("Training script for AFFormer-tiny model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="afformer-tiny-training", help="Experiment ID for Weights & Biases")
    return parser

def unfreeze_layers(model, epoch):
    if epoch == 8:  # epoch 9
        print("Unfreezing encoder4 (deepest block)")
        for param in model.encoder4.parameters():
            param.requires_grad = True
    elif epoch == 10:  # epoch 11
        print("Unfreezing encoder3")
        for param in model.encoder3.parameters():
            param.requires_grad = True
    elif epoch == 12:  # epoch 13
        print("Unfreezing encoder2")
        for param in model.encoder2.parameters():
            param.requires_grad = True
    elif epoch == 14:  # epoch 15
        print("Unfreezing encoder1")
        for param in model.encoder1.parameters():
            param.requires_grad = True
    elif epoch == 16:  # epoch 17
        print("Unfreezing input_layer (stem)")
        for param in model.input_layer.parameters():
            param.requires_grad = True

def main(args):
    wandb.init(project="5lsm0-cityscapes-segmentation", name=args.experiment_id, config=vars(args))
    
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = Compose([
        ToImage(),
        Resize((512, 512)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ])
    
    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transforms=transform)

    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=transform)
    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = AFFormerTiny(
        in_channels=3,
        n_classes=19).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

    best_valid_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        unfreeze_layers(model, epoch)

        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item(), "epoch": epoch + 1}, step=epoch * len(train_dataloader) + i)
        
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1).unsqueeze(1)
                    labels = labels.unsqueeze(1)
                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)
                    wandb.log({"predictions": [wandb.Image(make_grid(predictions.cpu(), nrow=8))], "labels": [wandb.Image(make_grid(labels.cpu(), nrow=8))]}, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({"valid_loss": valid_loss}, step=(epoch + 1) * len(train_dataloader) - 1)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(output_dir, f"best_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pth"))
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr}, step=(epoch + 1) * len(train_dataloader) - 1)

    print("Training complete!")
    torch.save(model.state_dict(), os.path.join(output_dir, f"final_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pth"))
    wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)