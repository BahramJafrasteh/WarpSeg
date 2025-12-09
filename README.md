# WarpSeg: Registration-Based Semi-Supervised Brain Segmentation

**WarpSeg** is a data-efficient deep learning framework for 3D brain MRI segmentation. By leveraging registration-based label warping and synthetic data generation, it achieves high-performance segmentation of detailed brain structures using limited manual annotations.

> **Note:** This method is also available as a ready-to-use plugin for the **MELAGE** software suite.

![WarpSeg Architecture](assets/figure_main.jpg)

## 🧠 Project Overview

This project addresses the challenge of segmenting detailed brain sub-regions when fully annotated data is scarce. It utilizes a **semi-supervised, dual-decoder architecture** that simultaneously learns coarse tissue classification and fine-grained anatomical parcellation.

### Key Features
* **Data-Efficient:** Trains effectively using limited ground truth by leveraging unlabelled data via registration.
* **Dual-Decoder Architecture:**
    * **Coarse Decoder:** Segments 6 major tissue classes (Background, GM, WM, Ventricles, CSF, Deep Gray Matter).
    * **Fine Decoder:** Segments 30 detailed anatomical regions.
* **Synthetic Data Augmentation:** Utilizes **3,000 synthetic images** generated via Latent Diffusion Models (LDM) to expand training variance.
* **MELAGE Compatible:** Designed to integrate seamlessly as a plugin for MELAGE.

---

## 🔬 Methodology

Our approach combines unsupervised deformable registration with semi-supervised segmentation.

### 1. Registration-Based Label Warping
We utilize **VoxelMorph**, a learning-based registration model, to align unlabelled images to a standard **MNI Space Atlas**. By inverting the deformation fields, we warp the detailed Atlas labels onto the unlabelled training images. This creates a massive corpus of "pseudo-labeled" data.

### 2. Hierarchical Multi-Decoder Network
The segmentation network is designed to handle the noise inherent in pseudo-labels using a hierarchical approach:
* **Shared Encoder:** Extracts multi-scale features from the input MRI.
* **Coarse Decoder:** Predicts robust high-level tissue masks.
* **Fine Decoder:** Predicts detailed anatomical structures.
* **Hierarchical Loss:** The model uses a specialized aggregation loss where the fine-grained predictions are summed and checked against the coarse ground truth. This guides the fine decoder even when detailed labels are noisy or missing.

### 3. Training Data
The model was trained on a massive, diverse dataset aligned to MNI space:
* **Real Data:** [IXI Dataset](https://brain-development.org/ixi-dataset/)
* **Synthetic Data:** 3,000 brain MRI images generated using Latent Diffusion Models (LDM).

---

## 📊 Performance & Benchmarks

We extensively evaluated the model on the **IXI** and **OASIS** datasets. We benchmarked performance against **FastSurfer** and **SynthSeg**.

### Comparative Results
We performed three specific comparisons using the Dice Similarity Coefficient (DSC):
1.  **WarpSeg vs. FastSurfer** (Using FastSurfer as Ground Truth)
2.  **WarpSeg vs. SynthSeg**
3.  **SynthSeg vs. FastSurfer** (Baseline inter-method variability)

As shown below, WarpSeg achieves competitive accuracy across major brain structures, robustly handling variability in input scans.

![Dice Score Comparison for IXI](IXI_dice.png)
*(Refer to `IXI_dice.pdf` in the repository for the high-resolution vector version of this plot)*


![Dice Score Comparison for OASIS dataset](OASIS_dice.png)
*(Refer to `OASIS_dice.pdf` in the repository for the high-resolution vector version of this plot)*

---

## 🛠️ Installation

### Prerequisites
* Python 3.8+
* NVIDIA GPU with CUDA support

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/BahramJafrasteh/WarpSeg.git](https://github.com/BahramJafrasteh/WarpSeg.git)
    cd WarpSeg-Brain
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

### 1. Inference (Prediction)
To segment a new brain MRI scan using the pre-trained model:

```bash
python test.py --input-image path/to/mri.nii.gz \
               --output-dir results/ \
               --model-path chkpts/best_model.pth

```
* **Output:** The script will generate two NIfTI files: `_seg_ours.nii.gz` (30 regions) and `_tissue_seg_ours.nii.gz` (6 tissues).

### 2. Training

To retrain the model on your own dataset:

**Step A: Train Registration (Optional)**
If you need to generate new pseudo-labels using VoxelMorph:
```bash
python train_reg.py --batchSize 1 --lr 1e-4 --lambda 0.01
```

**Step B: Train Segmentation To train the main dual-decoder network:**
```bash
python train.py --batchSize 24 --lr 0.001 --enc 16 32 32 32 --dec 32 32 32 32 32 16 16
```

## 📄 Citation

If you use this code, methodology, or the MELAGE plugin in your research, please cite this repository and the following paper:

> Jafrasteh, B., Lubián-López, S. P., & Benavente-Fernández, I. (2023). MELAGE: A purely Python-based Neuroimaging Software (Neonatal). arXiv preprint arXiv:2309.07175
> Jafrasteh, Bahram, et al. "MGA-Net: A novel mask-guided attention neural network for precision neonatal brain imaging." NeuroImage 300 (2024): 120872.






