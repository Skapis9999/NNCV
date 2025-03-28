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

# # Mapping class IDs to train IDs
# id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
# def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
#     return label_img.apply_(lambda x: id_to_trainid[x])

# # Mapping train IDs to color
# train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
# train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

# def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
#     batch, _, height, width = prediction.shape
#     color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

#     for train_id, color in train_id_to_color.items():
#         mask = prediction[:, 0] == train_id

#         for i in range(3):
#             color_image[:, i][mask] = color[i]

#     return color_image


# def get_args_parser():

#     parser = ArgumentParser("Training script for a PyTorch U-Net model")
#     parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
#     parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
#     parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
#     parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
#     parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
#     parser.add_argument("--experiment-id", type=str, default="peak-training-1", help="Experiment ID for Weights & Biases")

#     return parser

# def main(args):
#     # Initialize wandb for logging
#     wandb.init(
#         project="5lsm0-cityscapes-segmentation",  # Project name in wandb
#         name=args.experiment_id,  # Experiment name in wandb
#         config=vars(args),  # Save hyperparameters
#     )

#     # Create output directory if it doesn't exist
#     output_dir = os.path.join("checkpoints", args.experiment_id)
#     os.makedirs(output_dir, exist_ok=True)

#     # Set seed for reproducability
#     # If you add other sources of randomness (NumPy, Random), 
#     # make sure to set their seeds as well
#     torch.manual_seed(args.seed)
#     torch.backends.cudnn.deterministic = True

#     # Define the device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Define the transforms to apply to the data
#     transform = Compose([
#         ToImage(),
#         Resize((256, 256)),
#         ToDtype(torch.float32, scale=True),
#         Normalize((0.5,), (0.5,)),
#     ])

#     # Load the dataset and make a split for training and validation
#     train_dataset = Cityscapes(
#         args.data_dir, 
#         split="train", 
#         mode="fine", 
#         target_type="semantic", 
#         transforms=transform
#     )
#     valid_dataset = Cityscapes(
#         args.data_dir, 
#         split="val", 
#         mode="fine", 
#         target_type="semantic", 
#         transforms=transform
#     )

#     train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
#     valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

#     train_dataloader = DataLoader(
#         train_dataset, 
#         batch_size=args.batch_size, 
#         shuffle=True,
#         num_workers=args.num_workers
#     )
#     valid_dataloader = DataLoader(
#         valid_dataset, 
#         batch_size=args.batch_size, 
#         shuffle=False,
#         num_workers=args.num_workers
#     )

# # Load a pre-trained SAM model
# sam_checkpoint = "sam_vit_h_4b8939.pth"  # Ensure you have the correct model checkpoint
# model_type = "vit_h"  # Options: vit_b, vit_l, vit_h

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
# predictor = SamPredictor(sam)

# criterion = nn.CrossEntropyLoss(ignore_index=255)  # Adjust if needed

# optimizer = AdamW(sam.parameters(), lr=0.0001)

# def fine_tune_sam(args):
#     wandb.init(project="sam-cityscapes", name=args.experiment_id, config=vars(args))

#     train_dataset = Cityscapes(
#         args.data_dir,
#         split="train",
#         mode="fine",
#         target_type="semantic",
#         transforms=Compose([
#             ToImage(),
#             Resize((1024, 1024)),  # SAM prefers high-res images
#             ToDtype(torch.float32, scale=True),
#             Normalize((0.5,), (0.5,)),
#         ])
#     )

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

#     best_valid_loss = float("inf")

#     for epoch in range(args.epochs):
#         print(f"Epoch {epoch+1}/{args.epochs}")

#         sam.train()
#         for i, (images, labels) in enumerate(train_loader):
#             images, labels = images.to(device), labels.to(device)
            
#             # Get predictions from SAM
#             predictor.set_image(images[0].permute(1, 2, 0).cpu().numpy())  # Convert tensor to numpy for SAM
#             masks, scores, _ = predictor.predict(
#                 point_coords=None, point_labels=None, multimask_output=True
#             )

#             # Convert predictions to PyTorch tensor
#             predicted_masks = torch.tensor(masks, dtype=torch.float32, device=device)

#             # Compute loss
#             loss = criterion(predicted_masks, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             wandb.log({"train_loss": loss.item(), "epoch": epoch+1}, step=epoch * len(train_loader) + i)

#         print(f"Epoch {epoch+1} completed.")

#     print("Fine-tuning complete!")
#     torch.save(sam.state_dict(), os.path.join("checkpoints", "fine_tuned_sam.pth"))
#     wandb.finish()

# if __name__ == "__main__":
#     parser = get_args_parser()
#     args = parser.parse_args()
#     fine_tune_sam(args)    


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