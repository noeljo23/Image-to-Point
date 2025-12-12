"""
Evaluation Script - Uses the actual test split
Usage: python evaluate.py
"""

import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from model import PointCloudNet
from utils import PCDataset

# ============ CONFIGURATION ============
MODEL_PATH = "pc1024_three.pth"
NUM_SAMPLES = None  # None = use all test samples
BATCH_SIZE = 1
# =======================================


def compute_emd(pred, target):
    """Compute Earth Mover's Distance using Hungarian algorithm."""
    dist_matrix = cdist(pred, target, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    emd = dist_matrix[row_ind, col_ind].mean()
    return emd


def chamfer_distance_np(pred, target):
    """Compute Chamfer Distance between two point clouds."""
    # pred to target
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
    dist1, _ = nbrs.kneighbors(pred)

    # target to pred
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pred)
    dist2, _ = nbrs.kneighbors(target)

    cd = np.mean(dist1) + np.mean(dist2)
    return cd, dist1.flatten(), dist2.flatten()


def compute_fscore(dist1, dist2, threshold=0.01):
    """Compute F-Score."""
    precision = np.mean(dist1 < threshold)
    recall = np.mean(dist2 < threshold)

    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return fscore


def main():
    print("=" * 50)
    print("RGB2Point Model Evaluation")
    print("Using official test split")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = PointCloudNet(
        num_views=1,
        point_cloud_size=1024,
        num_heads=4,
        dim_feedforward=2048
    )

    checkpoint = torch.load(
        MODEL_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Load test dataset using YOUR PCDataset class
    print("\nLoading test dataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    test_dataset = PCDataset(stage="test", transform=transform)

    if NUM_SAMPLES:
        # Limit samples if specified
        test_dataset.data = test_dataset.data[:NUM_SAMPLES]

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    # Evaluation metrics
    cd_scores = []
    emd_scores = []
    f_scores_001 = []
    f_scores_01 = []
    inference_times = []

    print(f"\nEvaluating...")
    print("-" * 50)

    for images, gt_pc, names in tqdm(test_loader, desc="Evaluating"):
        try:
            images = images.to(device)

            # Run inference with timing
            start_time = time.time()
            with torch.no_grad():
                pred_pc = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Get numpy arrays
            pred_np = pred_pc[0].cpu().numpy()
            gt_np = gt_pc[0].cpu().numpy()

            # Compute Chamfer Distance
            cd, dist1, dist2 = chamfer_distance_np(pred_np, gt_np)
            cd_scores.append(cd)

            # Compute EMD
            emd = compute_emd(pred_np, gt_np)
            emd_scores.append(emd)

            # Compute F-Score at different thresholds
            fscore_001 = compute_fscore(dist1, dist2, threshold=0.001)
            fscore_01 = compute_fscore(dist1, dist2, threshold=0.01)
            f_scores_001.append(fscore_001)
            f_scores_01.append(fscore_01)

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Samples evaluated: {len(cd_scores)}")
    print("-" * 50)
    print(
        f"Chamfer Distance (CD):    {np.mean(cd_scores):.6f}  (lower is better)")
    print(
        f"Earth Mover Distance:     {np.mean(emd_scores):.6f}  (lower is better)")
    print(
        f"F-Score @ 0.001:          {np.mean(f_scores_001):.4f}    (higher is better)")
    print(
        f"F-Score @ 0.01:           {np.mean(f_scores_01):.4f}    (higher is better)")
    print(f"Inference Time:           {np.mean(inference_times):.4f} sec")
    print("=" * 50)

    # Also show scaled CD (papers often report CD * 1000)
    print(f"\nCD x 1000:                {np.mean(cd_scores) * 1000:.3f}")
    print(f"EMD x 1000:               {np.mean(emd_scores) * 1000:.3f}")

    # Save results
    with open("evaluation_results.txt", "w") as f:
        f.write("RGB2Point Evaluation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Samples: {len(cd_scores)}\n")
        f.write(f"Chamfer Distance: {np.mean(cd_scores):.6f}\n")
        f.write(f"EMD: {np.mean(emd_scores):.6f}\n")
        f.write(f"F-Score@0.001: {np.mean(f_scores_001):.4f}\n")
        f.write(f"F-Score@0.01: {np.mean(f_scores_01):.4f}\n")
        f.write(f"Inference Time: {np.mean(inference_times):.4f} sec\n")
        f.write(f"\nCD x 1000: {np.mean(cd_scores) * 1000:.3f}\n")
        f.write(f"EMD x 1000: {np.mean(emd_scores) * 1000:.3f}\n")

    print("\nResults saved to evaluation_results.txt")


if __name__ == "__main__":
    main()
