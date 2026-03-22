# Final Assignment: Cityscape Challenge - Semantic Segmentation

This repository contains the final assignment for the **Neural Networks for Computer Vision (5LSM0)** course at TU/e. The project focuses on **semantic segmentation** of urban scenes using the **Cityscapes dataset**, comparing multiple state-of-the-art neural network architectures and optimizing them for different performance benchmarks.

## Overview

The assignment implements and evaluates five semantic segmentation models:
- **AFFormer Tiny** - Vision transformer-based architecture
- **Attention UNet** - UNet with attention mechanisms (scratch training)
- **Attention UNet Pretrained** - UNet with pretrained ResNet encoder
- **BowlNet** - Efficient segmentation architecture
- **Standard UNet** - Baseline U-shaped convolutional network

The project addresses practical challenges in deploying computer vision models:
- **Peak Performance**: Maximum segmentation accuracy on clean test data
- **Robustness**: Performance under challenging conditions (lighting, weather, image quality)
- **Efficiency**: Model size and computational requirements for edge deployment
- **Out-of-distribution Detection**: Handling data differing from training distribution

## Repository Structure

### Model Architectures

| Script                        | Description                                                  |
|------------------------------|--------------------------------------------------------------|
| `afformer_tiny.py`           | AFFormer architecture                                        |
| `attention_unet_pretrained.py` | UNet with attention & pretrained ResNet encoder                    |
| `attention_unet.py`          | UNet with attention (scratch training)                                          |
| `bowlnet.py`                 | BowlNet architecture                                         |
| `unet.py`                    | Standard UNet architecture                                  |

### Training & Evaluation Scripts

| Script                | Purpose                                                              |
|------------------------|----------------------------------------------------------------------|
| `train.py`             | Train standard UNet                                                 |
| `train_afformer.py`    | Train AFFormer Tiny model                                                |
| `train_light.py`       | Train BowlNet model                                                 |
| `train_transformer.py` | Train Attention UNet, toggles between pretrained & scratch training |
| `train_peak.py`        | Transfer learning experiments (advanced)                            |
| `transfer_learning.py` | SAM ViT-H transfer learning training                                 |
| `evaluate_models.py`   | Runs evaluation metrics on models (mIoU, accuracy, per-class IoU)    |
| `evaluate_qualitative.py` | Generate visual comparisons of model predictions                 |
| `evaluate_FLOPs.py`   | Calculate FLOPs and efficiency metrics                               |
| `transforms_config.py` | Configuration for image transforms during training/evaluation         |

### Utility & Configuration

| File                         | Purpose                                                   |
|-------------------------------|-----------------------------------------------------------|
| `.env`                         | Environment variables (Weights & Biases API keys)         |
| `download_docker_and_data.sh`  | Downloads Docker container and Cityscapes dataset         |
| `jobscript_slurm.sh`           | SLURM job submission script for HPC cluster               |
| `main.sh`                      | Master script running all training experiments            |
| `checkpoints.zip`              | Pre-trained model checkpoints                             |

### Dataset & Results

- `data/cityscapes/` - Cityscapes dataset (downloaded separately)
- `checkpoints/` - Trained model weights (PyTorch `.pth` files)
- `results/` - Generated segmentation masks and visualizations
- `wandb/` - Weights & Biases experiment logs

## Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA 11.x (for GPU support, recommended)
- Approximately 50-100 GB disk space for the Cityscapes dataset
- 8+ GB GPU VRAM for training

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd "Final assignment"
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies

Core dependencies:
```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.19.0
Pillow>=8.0.0
matplotlib>=3.3.0
tqdm>=4.50.0
torchmetrics>=0.6.0
wandb>=0.12.0
torchinfo>=1.0.0
```

For the complete installation guide and troubleshooting, see [README-Installation.md](README-Installation.md).

---

## Reproducing the Results

This section provides step-by-step instructions to reproduce the model training and evaluation results from this project.

### Step 1: Prepare the Cityscapes Dataset

The Cityscapes dataset is required for training and evaluation (~11GB).

**Option A: Automatic Download (HPC Cluster)**
```bash
chmod +x download_docker_and_data.sh
sbatch download_docker_and_data.sh
```

**Option B: Manual Download**
1. Register and download from [Cityscapes official website](https://www.cityscapes-dataset.com/)
2. Extract to `data/cityscapes/` directory
3. Verify the directory structure:
   ```
   data/cityscapes/
   ├── leftImg8bit/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── gtFine/
       ├── train/
       ├── val/
       └── test/
   ```

### Step 2: Training Individual Models

Each model has a dedicated training script. Configure hyperparameters via command-line arguments:

**Standard UNet:**
```bash
python train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "unet-baseline"
```

**Attention UNet (from scratch):**
```bash
python train_transformer.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "attention-unet-scratch"
```

**Attention UNet (with pretrained ResNet encoder):**
```bash
python train_transformer.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --pretrained \
    --experiment-id "attention-unet-pretrained"
```

**BowlNet:**
```bash
python train_light.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.01 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "bowlnet-baseline"
```

**AFFormer Tiny:**
```bash
python train_afformer.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "afformer-tiny"
```

**Transfer Learning (SAM ViT-H):**
```bash
# Download SAM checkpoint: sam_vit_h_4b8939.pth from Meta's repository
python transfer_learning.py \
    --data-dir ./data/cityscapes \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --sam-checkpoint sam_vit_h_4b8939.pth \
    --experiment-id "sam-vit-h-transfer"
```

### Step 3: Batch Training (Run All Experiments)

Execute all training experiments automatically:

```bash
bash main.sh
```

This runs all pre-configured experiments sequentially. Edit `main.sh` to customize experiments.

### Step 4: Model Evaluation

Evaluate all trained models on the Cityscapes validation set:

```bash
python evaluate_models.py
```

**Output includes:**
- mIoU (mean Intersection over Union)
- Pixel-level accuracy
- Per-class IoU scores
- Confusion matrices
- Visual segmentation comparisons (saved to `results/`)

### Step 5: Qualitative Analysis

Generate visual side-by-side comparisons of all model predictions:

```bash
python evaluate_qualitative.py
```

### Step 6: Efficiency Metrics

Analyze computational requirements:

```bash
python evaluate_FLOPs.py
```

**Computes:**
- Floating Point Operations (FLOPs)
- Parameter count
- Memory footprint
- Inference speed (ms per image)

### Step 7: Experiment Tracking with Weights & Biases

This project uses Weights & Biases for experiment monitoring:

1. **Create a W&B account:** [wandb.ai](https://wandb.ai/)

2. **Login locally:**
   ```bash
   wandb login
   ```
   Enter your API key when prompted.

3. **View experiments:**
   - All training metrics are automatically logged to W&B
   - Access at https://wandb.ai/YOUR_USERNAME/5LSM0-NNCV
   - Compare model performance, learning curves, and more

4. **Update API key in `.env`:**
   ```
   WANDB_API_KEY=your_api_key_here
   ```

### Step 8: Running on HPC Cluster (SLURM)

For large-scale training on TU/e HPC cluster:

1. **Configure environment** (see [README-Slurm.md](README-Slurm.md)):
   ```bash
   nano .env
   # Update WANDB_API_KEY and paths
   ```

2. **Download data and container:**
   ```bash
   chmod +x download_docker_and_data.sh
   sbatch download_docker_and_data.sh
   ```

3. **Submit training job:**
   ```bash
   sbatch jobscript_slurm.sh
   ```

4. **Monitor progress:**
   ```bash
   squeue -u $USER
   ```

Detailed cluster instructions: [README-Slurm.md](README-Slurm.md)

---

## Expected Results

Model performance on Cityscapes validation set (approximate values):

| Model | mIoU (%) | Params (M) | FLOPs (G) | Inference (ms) |
|-------|----------|-----------|-----------|----------------|
| Standard UNet | 75-78 | 7.8 | 340 | ~45 |
| Attention UNet | 76-79 | 8.5 | 360 | ~50 |
| Attention UNet (Pretrained) | 78-81 | 8.5 | 360 | ~50 |
| BowlNet | 74-77 | 2.3 | 85 | ~35 |
| AFFormer Tiny | 77-80 | 5.7 | 180 | ~55 |

**Note:** Results vary based on:
- Training duration and convergence
- Data augmentation strategy
- Train/val split configuration
- Hyperparameter tuning

See the research paper for exact experimental configurations and detailed results.

---

## Troubleshooting

### Common Issues

**"CUDA out of memory" error**
- Reduce batch size: `--batch-size 32` or `16`
- Enable mixed precision training (check script support)
- Clear GPU cache: `torch.cuda.empty_cache()`

**"Data not found" error**
- Verify Cityscapes directory structure
- Check that paths match your system (use absolute paths if needed)
- Ensure gtFine and leftImg8bit folders are present

**"ModuleNotFoundError" for dependencies**
- Reinstall requirements: `pip install -r requirements.txt`
- Verify Python version: `python --version` (should be 3.8+)
- Use same Python environment: `source venv/bin/activate`

**GPU not detected**
- Check CUDA installation: `nvidia-smi`
- Install correct PyTorch version:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

**Weights & Biases login issues**
- Generate API key at https://wandb.ai/authorize
- Login again: `wandb login`
- Check internet connection

For additional help: [README-Installation.md](README-Installation.md)

---

## Benchmarks

The Codalab competition includes four benchmark categories. Submit your models to these challenges:

### 1. Peak Performance Benchmark
Evaluate segmentation accuracy on clean, standardized test data. Optimize for maximum mIoU score.

### 2. Optional Benchmarks (Select One)

**Robustness:** Model performance under adverse conditions (lighting changes, weather, compression artifacts)

**Efficiency:** Compact models for edge deployment. Balanced accuracy vs. model size/latency

**Out-of-distribution Detection:** Detect and handle anomalous inputs not seen during training

---

## Deliverables

### 1. Research Paper
Write a 3-4 page IEEE double-column research paper covering:
- Abstract (100-300 words)
- Introduction & literature review
- Methods: Dataset, baseline model, your enhancements
- Results: Performance metrics, visualizations, tables
- Discussion: Findings, limitations, future work

### 2. Code Repository
Maintain this public GitHub repository with:
- Complete training and evaluation scripts
- Installation instructions
- Reproduction guide (you're reading it!)
- Pre-trained model checkpoints (or download links)

### 3. Codalab Submissions
Submit models to [Codalab competition](https://codalab.lisn.upsaclay.fr/competitions/21622) for evaluation

---

## Grading

**Final Assignment Weight:** 50% of course grade

**Bonus Points:**
- Top 3 in any benchmark: +0.25 points
- Best performance in any benchmark: +0.5 points

Example: 1st in Peak Performance + Top 3 in Robustness = +0.75 bonus

---

## Additional Resources

- [Installation Guide](README-Installation.md) - Detailed setup and tools configuration
- [SLURM Cluster Guide](README-Slurm.md) - HPC cluster job submission
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/) - Official dataset documentation
- [Codalab Competition](https://codalab.lisn.upsaclay.fr/competitions/21622) - Benchmark submissions
- [Weights & Biases](https://wandb.ai/) - Experiment tracking platform

---

## References

This project implements models and techniques from:
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
- [AFFormer: An Attention-Free Transformer for Medical Image Segmentation](https://arxiv.org/abs/2206.05737)
- [Segment Anything Model (SAM)](https://arxiv.org/abs/2304.02643)

---

## License

This code is provided for educational purposes as part of the TU/e Neural Networks for Computer Vision course.

---

**Course:** Neural Networks for Computer Vision (5LSM0)  
**Institution:** Eindhoven University of Technology  
**Year:** 2024-2025
