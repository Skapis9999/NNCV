# Final Assignment: Cityscape Challenge  

## 

This repository contains the final assignment for the **NNCV** course at TU/e, focused on comparing multiple semantic segmentation models on the Cityscapes dataset. The goal is to perform inference using several models and visualize their predictions side-by-side.

## Models Included

- **AFFormer Tiny**
- **Attention UNet (Pretrained & Finetuned versions)**
- **BowlNet**
- **Standard UNet**

## Prepare the folders:

- Place the trained model checkpoints (.pth files) inside their respective subfolders in checkpoints/.
-  Place all input images (for inference) inside data/cityscapes/.

## Student Info

- Codalab Username: skapis9999

- TU/e Email: c.skapetis@student.tue.nl

## Content

### Model Architectures:

| Script                        | Description                                                  |
|------------------------------|--------------------------------------------------------------|
| `afformer_tiny.py`           | AFFormer architecture                                        |
| `attention_unet_pretrained.py` | UNet with attention & pretrained encoder                    |
| `attention_unet.py`          | UNet with attention                                          |
| `bowlnet.py`                 | BowlNet architecture                                         |
| `unet.py`                    | Standard UNet architecture   

### Images:

| File                          | Purpose                                                   |
|-------------------------------|-----------------------------------------------------------|
| `800px-Edsger_Wybe_Dijkstra.jpg` | Easter Egg image for initial testing                     |
| `Comparison_correct.png`        | Comparison image used in the final report
| `Comparison_correct.png`        | Initial Image that was supposed to be used in the final report  

### .sh files:

| Script                         | Description                                                   |
|--------------------------------|---------------------------------------------------------------|
| `download_docker_and_data.sh`  | Downloads the container and dataset                          |
| `jobscript_slurm.sh`           | SLURM job script (manages GPUs, run time, and calls `main.sh`) |
| `main.sh`                      | Main training script with multiple experiment configurations |

### pth models:

- sam_vit_h_4b8939.pth -> model used for trasfer learning. (It is not uploaded on github due to size limitations.) You need to download it to run the transfer_learning.py file.
- checkpoints -> folder with all the checkpoints saved during the training sessions

### data:

- data -> folder with all the data. When you train the model it's the data downloaded by the .sh file. When you evaluate your model delete this folder or rename it and replace it with the 3 pictures you want to segment.

### Training scripts:

| Script                | Purpose                                                              |
|------------------------|----------------------------------------------------------------------|
| `train_afformer.py`    | Train AFFormer model                                                |
| `train_light.py`       | Train BowlNet model                                                 |
| `train_peak.py`        | Attempted training with transfer learning (did not succeed)         |
| `train_transformer.py` | Train Attention UNet, toggles between pretrained & scratch training |
| `train.py`             | Train standard UNet                                                 |
| `transfer_learning.py` | Transfer learning training (did not succeed)                        |

### Evaluation scripts:

| Script                   | Purpose                                              |
|--------------------------|------------------------------------------------------|
| `transforms_config.py`   | Contains transform configurations for evaluation     |
| `evaluate_FLOPs.py`      | Calculates FLOPs (can be run locally)               |
| `evaluate_models.py`     | Runs evaluation metrics on models   

### Requirments:

torch>=1.10
torchvision>=0.11
numpy
Pillow
matplotlib
tqdm


If you encounter any issues or have questions, feel free to reach out via TU/e email.

-----------------------------------------

Welcome to the **Cityscape Challenge**, the final project for this course!  

In this assignment, you'll put your knowledge of Neural Networks (NNs) for computer vision into action by tackling real-world problems using the **CityScapes dataset**. This dataset contains large-scale, high-quality images of urban environments, making it perfect for tasks like **semantic segmentation** and **object detection**.  

This challenge is designed to push your skills further, focusing on practical and often under-explored issues crucial for deploying computer vision models in real-world scenarios.  

---

## Benchmarks  

The competition comprises four benchmarks, each targeting a specific aspect of model performance:  

1. **Peak performance**  
   This benchmark evaluates your model's segmentation accuracy on a clean, standardized test set. Your goal is to achieve the highest segmentation scores here. **Everyone should submit a model to this benchmark optimized for maximum performance**. However, it's crucial to implement changes thoughtfully and be able to justify them in your research paper. Ultimately, the focus should be on the scientific contributions of your adaptations rather than solely aiming for the highest score.

The following benchmarks 2–4 are optional, and **you should select one** to compare against the Peak Performance benchmark. This allows you to analyze how your model performs under different conditions and gain deeper insights beyond just optimizing for the highest score.

2. **Robustness**  
   This benchmark tests how well your model performs under challenging conditions, such as changes in lighting, weather, or image quality. Consistency is key in this category.  

3. **Efficiency**  
   Practical applications often require compact models. This benchmark emphasizes creating smaller models that maintain acceptable performance. It’s particularly relevant for edge devices where large models are infeasible.  

4. **Out-of-distribution detection**  
   Models often encounter data that differs from the training distribution, leading to unreliable predictions. This benchmark evaluates your model's ability to detect and handle such out-of-distribution samples.  

> **IMPORTANT NOTE**: A fifth benchmark, **Baseline**, will be available on the competition server, and all participants must submit a baseline model. The training code for this model is already provided. This benchmark serves two purposes: ensuring that everyone is familiar with working on an HPC cluster and providing a reference point for evaluating the impact of different adaptations in your other benchmark submissions. The Baseline benchmark will close on **Sunday, March 16, at 11:59 P.M. (GMT+1)**. To avoid last-minute issues, start preparing your submission early. This will also give you time to ask questions during the scheduled computer classes if needed.

---

## Deliverables  

Your final submission will consist of the following:  

### 1. Research paper  
Write a **3-4 page research paper** in [IEEE double-column format](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn), addressing (at least) the following:  

- **Abstract**: Summarize the current problems, your key steps for addressing them and your main findings in about 100-300 words.
- **Introduction**: Present the problem, challenges, and potential solutions based on existing literature.  
- **Methods**: Describe your dataset(s), outline the baseline approach using an off-the-shelf segmentation model and define the enhancements you made for the specific benchmarks you participated.  
- **Results**: Show and describe your results based on performance metrics and examples. Use figures and tables to support your findings. 
- **Discussion**: Discuss the impact and potential of your main findings. Also discuss limitations and suggest future improvements.

> **Submission**: Submit your paper as a PDF document via **Canvas**.

The paper will be graded based on clarity, experimental design, insight, and originality.  

### 2. Code repository  
Push all relevant code to a **public GitHub repository** with a README.md file detailing:  
- Required libraries and installation instructions.  
- Steps to run your code.  
- Your Codalab username and TU/e email address for correct mapping across systems.  

### 3. Codalab submissions  
Submit your model for evaluation to the [**Codalab challenge server**](https://codalab.lisn.upsaclay.fr/competitions/21622), which includes four benchmark test sets.
> Make sure to [sign up](https://codalab.lisn.upsaclay.fr/accounts/signup/?next=/) if you do not have an account on Codalab yet.

---

## Grading and Bonus Points  

The final assignment accounts for **50% of your course grade**. Additionally, bonus points are available:  

- **Top 3 in any benchmark**: +0.25 to your final assignment grade.  
- **Best performance in any benchmark**: +0.5 to your final assignment grade.  

For example, achieving the best performance in 'Peak Performance' and a top 3 spot in another benchmark will earn you a 0.75 bonus.  

> **Note**: The bonus is optional. A great report with an innovative solution that doesn't rank highly can still earn a perfect score (10).  

---

## Important Notes  

- Ensure a proper **train-validation split** of the CityScapes dataset.  
- Training your model may take multiple hours; plan accordingly.  
- Use ideas from literature but remember to **cite all sources**. Plagiarism will not be tolerated.  
- For questions or challenges, use the **Discussions** section of this repository to collaborate with peers.  

---

We wish you the best of luck in this challenge and are excited to see the innovative solutions you develop! 🚀