import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype
import wandb
from argparse import ArgumentParser
from segment_anything import sam_model_registry, SamPredictor

# Load SAM Model
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Ensure that I downloaded this or something else
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
sam.train()  # Set SAM to training mode

# Use the predictor for fine-tuning
predictor = SamPredictor(sam)

# Define loss function
criterion = nn.CrossEntropyLoss(ignore_index=255) # Ignore the void class

# Optimizer
optimizer = AdamW(sam.parameters(), lr=args.lr)

def fine_tune_sam(args):
    wandb.init(project="sam-cityscapes", name=args.experiment_id, config=vars(args))

    transform = Compose([
        ToImage(),
        Resize((1024, 1024)),  # SAM prefers high-res images
        ToDtype(torch.float32, scale=True),
        Normalize((0.5,), (0.5,)),
    ])

    # Load train and validation datasets
    train_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=transform)
    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_valid_loss = float("inf")
    model_save_path = os.path.join("checkpoints", f"best_sam_{args.experiment_id}.pth")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Training
        sam.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Get predictions from SAM
            predictor.set_image(images[0].permute(1, 2, 0).cpu().numpy())  # Convert tensor to numpy for SAM
            masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, multimask_output=True)

            # Convert predictions to PyTorch tensor
            predicted_masks = torch.tensor(masks, dtype=torch.float32, device=device)

            # Compute loss
            loss = criterion(predicted_masks, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            wandb.log({"train_loss": loss.item(), "epoch": epoch+1}, step=epoch * len(train_loader) + i)

        # Validation
        sam.eval()
        valid_loss = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                predictor.set_image(images[0].permute(1, 2, 0).cpu().numpy())
                masks, _, _ = predictor.predict(point_coords=None, point_labels=None, multimask_output=True)
                predicted_masks = torch.tensor(masks, dtype=torch.float32, device=device)

                loss = criterion(predicted_masks, labels)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        wandb.log({"valid_loss": valid_loss, "epoch": epoch+1})

        print(f"Epoch {epoch+1} | Train Loss: {train_loss / len(train_loader):.4f} | Valid Loss: {valid_loss:.4f}")

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(sam.state_dict(), model_save_path)
            print(f"New best model saved at {model_save_path}")

    print("Fine-tuning complete!")
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--experiment-id", type=str, default="sam-finetune")

    args = parser.parse_args()
    fine_tune_sam(args)