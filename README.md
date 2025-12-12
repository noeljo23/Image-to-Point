# Image-to-Point

A Vision Transformer-based deep learning model for generating 3D point clouds from single RGB images. This project achieves real-time inference (~5.7ms) while maintaining competitive reconstruction quality.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Dependencies](#dependencies)
- [Citation](#citation)

## 🎯 Overview

Image-to-Point generates 1024-point 3D point clouds from single RGB images using:
- **Vision Transformer (ViT)** backbone for robust feature extraction
- **Cross-Feature Integration (CFI)** module with multi-head self-attention
- **Point Cloud Generation Module (GPM)** for direct 3D coordinate regression

### Key Features
- ⚡ **Real-time inference**: ~5.7ms per image (175+ FPS)
- 🎯 **Competitive accuracy**: CD = 0.0379, F-Score@0.01 = 0.2181
- 💾 **Memory efficient**: Frozen ViT backbone reduces training memory
- 🖥️ **Interactive demo**: Streamlit-based web application

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 4GB+ GPU memory

### Setup

```bash
# Clone the repository
git clone https://github.com/noeljo23/Image-to-Point.git
cd Image-to-Point

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
RGB2Point/
├── model.py           # Neural network architecture
│                      # - PointCloudNet: Main model class
│                      # - PointCloudGeneratorWithAttention: CFI + GPM modules
├── train.py           # Training script with distributed training support
│                      # - Uses Accelerate for multi-GPU training
│                      # - WandB integration for logging
├── evaluate.py        # Evaluation script
│                      # - Chamfer Distance, EMD, F-Score metrics
│                      # - Batch evaluation on test set
├── inference.py       # Single-image inference
│                      # - Load model and generate point cloud
├── utils.py           # Utility functions
│                      # - PCDataset class for ShapeNet
│                      # - Chamfer distance implementation
│                      # - Point cloud visualization
├── streamlit_app.py   # Interactive web demo
│                      # - Upload image → Generate 3D point cloud
│                      # - Real-time visualization with Plotly
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## 🚀 Usage

### Quick Inference

```python
from model import PointCloudNet
from utils import predict
import torch

# Load model
model = PointCloudNet(
    num_views=1,
    point_cloud_size=1024,
    num_heads=4,
    dim_feedforward=2048
)
model.load_state_dict(torch.load("pc1024_three.pth")["model"])
model.eval()

# Generate point cloud from image
predict(model, "input_image.jpg", "output.ply")
```

### Interactive Demo

```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

## 🏋️ Training

### Dataset Preparation

1. Download ShapeNet dataset
2. Generate rendered images using Blender
3. Sample point clouds (1024 points) from meshes
4. Update paths in `train.py` and `utils.py`:
   - `pc_base`: Path to point cloud directory
   - `img_base`: Path to rendered images directory
   - `list_file`: Path to train/test split files

### Training Command

```bash
# Single GPU
python train.py

# Multi-GPU with Accelerate
accelerate launch train.py
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| LR Scheduler | ReduceLROnPlateau |
| Loss Function | Bidirectional Chamfer Distance |
| Point Cloud Size | 1024 |

## 📊 Evaluation

```bash
python evaluate.py
```

### Evaluation Metrics

- **Chamfer Distance (CD)**: Average nearest-neighbor distance
- **Earth Mover's Distance (EMD)**: Optimal transport distance
- **F-Score**: Precision-recall at distance thresholds

## 📈 Results

### ShapeNet Test Set (1,517 samples)

| Metric | Value |
|--------|-------|
| Chamfer Distance | 0.0379 (×1000: 37.9) |
| Earth Mover's Distance | 0.0475 (×1000: 47.5) |
| F-Score @ 0.001 | 0.0003 |
| F-Score @ 0.01 | 0.2181 |
| Inference Time | 5.7 ms |

### Categories Evaluated
- Airplanes (02691156)
- Cars (02958343)
- Chairs (03001627)

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
accelerate>=0.20.0
scipy>=1.10.0
scikit-learn>=1.2.0
numpy>=1.24.0
Pillow>=9.5.0
tqdm>=4.65.0
open3d>=0.17.0
streamlit>=1.25.0
plotly>=5.15.0
wandb>=0.15.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🏗️ Model Architecture

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────┐
│  ViT-Base-16    │  Pretrained, Frozen
│  (768-dim out)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Aggregator    │  Linear: 768 → 4096
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CFI Module     │  Multi-Head Self-Attention
│  (H=4 or H=16)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPM Module     │  MLP: 4096 → 2048 → 2048 → 3072
│  (LeakyReLU)    │
└────────┬────────┘
         │
         ▼
Output Point Cloud (1024×3)
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [timm](https://github.com/huggingface/pytorch-image-models) for Vision Transformer implementation
- [ShapeNet](https://shapenet.org/) for the 3D model dataset
- [Accelerate](https://huggingface.co/docs/accelerate) for distributed training support

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Author:** Noel John  
**Institution:** Northeastern University  
**Project:** Deep Learning Course Final Project

---

**Note**: This project was developed as part of a Deep Learning course. The model weights and sample data are available upon request.
