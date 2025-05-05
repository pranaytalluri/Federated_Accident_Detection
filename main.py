"""
main.py

Entrypoint script to run federated training and evaluation pipeline:
  1. Initialize Weights & Biases (wandb) for experiment tracking
  2. Set random seeds for reproducibility
  3. Select compute device (MPS/CUDA/CPU)
  4. Load and partition video dataset across clients
  5. Initialize global model
  6. Execute federated rounds via run_federated_rounds
  7. Perform final model evaluation on each client's test set
"""

# === Standard library imports ===
import warnings  # To suppress non-critical warnings

# === Third-party imports ===
import wandb  # Weights & Biases for experiment logging and config

# === Local imports ===
from train import run_federated_rounds  # Federated training workflow
from evaluate import evaluate_model     # Final evaluation routine
from utils import (
    set_seed,             # Seed RNGs (Python, NumPy, PyTorch)
    select_device,        # Auto-select MPS/CUDA/CPU
    partition_clients,     # Split dataset into client shards
    gather_video_paths_and_labels  # Scan directories for videos + labels
)
from model import initialize_model  # Create model architectures

# Suppress warnings to keep console output focused on key logs
warnings.filterwarnings("ignore")

# === 1) Initialize W&B experiment ===
# Project name identifies grouping in wandb UI; name tags this run
wandb.init(
    project="FL_I3D_Training",
    name="run_federated_simulation",
    config={
        "rounds": 3,    # Number of federated rounds to run
        "lr": 1e-4       # Base learning rate for local training
    }
)

# === 2) Set random seeds for reproducibility ===
# Using a fixed seed ensures consistent data splits and training behavior
seed = 42
set_seed(seed)

# === 3) Select compute device ===
# Uses MPS if available on macOS, otherwise falls back to CPU/CUDA
device = select_device()

# === 4) Load and partition dataset ===
# Assumes base_dir contains 'accident' and 'no_accident' subfolders with .mp4 files
base_dir = "/Users/pranaytalluri/Downloads/Federated_Accident_Detection/Data"
all_files, all_labels = gather_video_paths_and_labels(base_dir)
# Partition into K clients with a train/val/test split per client
client_splits = partition_clients(
    files=all_files,
    labels=all_labels,
    K=2,          # Number of clients
    val_size=0.2  # Fraction of data per client reserved for validation
)

# === 5) Initialize global model ===
# Create a small 3D CNN for binary classification
global_model, device = initialize_model(
    num_classes=2,
    model_type="small3dcnn"
)
print("Running on:", device)

# === 6) Run federated training rounds ===
# Returns the updated global_model and its final parameters
global_model, averaged_params = run_federated_rounds(
    global_model=global_model,
    client_splits=client_splits,
    num_rounds=wandb.config.rounds,
    device=device,
)

# === 7) Final model evaluation ===
# Evaluate the global federated model on each client's test set
evaluate_model(
    global_model=global_model,
    averaged_params=averaged_params,
    client_splits=client_splits,
    device=device
)
