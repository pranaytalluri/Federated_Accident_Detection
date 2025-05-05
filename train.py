"""
train.py

Implements the core federated training loop:
  - Aggregates weight updates (deltas) from each client
  - Applies secure mask exchange for privacy
  - Updates the global model parameters
  - Logs metrics per round to Weights & Biases
"""

# === Standard library imports ===
import warnings  # To suppress non-critical warnings

# === Third-party imports ===
import torch              # PyTorch core
import wandb              # Experiment logging
import numpy as np        # Numerical operations for deltas and masks
from torch.utils.data import DataLoader  # Batching client datasets
from typing import Tuple, List  # Type hints for clarity
import flwr as fl         # Flower federated learning framework
from flwr.common import (
    ndarrays_to_parameters,  # Serialize NumPy arrays to FL parameters
    parameters_to_ndarrays   # Deserialize FL parameters to NumPy arrays
)

# === Local imports ===
from model import initialize_model  # Factory for model instantiation
from data import CADPClipDataset     # Dataset for video clips
from utils import get_model_parameters, set_model_parameters
from client import FLClient, SEED_STORE  # Custom client and seed store

# Suppress warnings for cleaner console output
warnings.filterwarnings("ignore")


def aggregate_metrics(results):
    """
    Compute average validation accuracy and loss across all clients.

    Args:
        results (List[Tuple[int, dict]]): List of (num_examples, metrics) per client.

    Returns:
        dict: {'accuracy': float, 'loss': float}
    """
    # Extract val_accuracy and val_loss from each client's metrics
    acc_values = [metrics.get("val_accuracy", 0.0) for _, metrics in results]
    loss_values = [metrics.get("val_loss", 0.0) for _, metrics in results]
    # Compute simple arithmetic mean
    return {
        "accuracy": sum(acc_values) / len(acc_values) if acc_values else 0.0,
        "loss": sum(loss_values) / len(loss_values) if loss_values else 0.0
    }


def run_federated_rounds(
    global_model: torch.nn.Module,
    client_splits: List[Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]],
    num_rounds: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, List]:
    """
    Execute `num_rounds` of federated training.

    1) Snapshot global weights before each round
    2) Clear any leftover seed store for privacy
    3) For each client:
         - Build data loaders (train, val, test)
         - Initialize a local model copy at the global state
         - Run client.fit() to get masked weight delta
    4) Aggregate (sum and average) masked deltas to compute new global update
    5) Log aggregated metrics and update the global model

    Args:
        global_model: The shared model to be trained.
        client_splits: Per-client (train/val/test) file and label splits.
        num_rounds: Total federated communication rounds.
        device: torch.device to run training (CPU/MPS/CUDA).

    Returns:
        Tuple of updated global_model and its final flat parameter list.
    """
    # Loop over federated rounds
    for rnd in range(num_rounds):
        print(f"\n🌐 Round {rnd+1} starting...")

        # 1) Take a snapshot of the current global model weights
        base_params = get_model_parameters(global_model)  # Returns list of NumPy arrays

        # 2) Reset the seed store for secure aggregation
        SEED_STORE.clear()

        # Preallocate accumulator for client masked deltas
        agg_masked = [np.zeros_like(p) for p in base_params]

        # Number of clients participating in this round
        num_clients = len(client_splits)
        # Collect per-client results for logging
        round_results = []

        # 3) Iterate over each client partition
        for client_id, split in enumerate(client_splits):
            print(f"\n🚀 Client {client_id} starting round {rnd+1}...")

            # Unpack file paths and labels: train, val, test
            train_f, train_l, val_f, val_l, test_f, test_l = split
            # Create Datasets for each subset
            train_ds = CADPClipDataset(train_f, train_l)
            val_ds = CADPClipDataset(val_f, val_l)
            test_ds = CADPClipDataset(test_f, test_l)

            # Create DataLoaders: small batch sizes suitable for DP
            train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

            # 3a) Initialize a fresh local model at the global weights
            client_model, _ = initialize_model(num_classes=2, model_type="small3dcnn")
            set_model_parameters(client_model, base_params)

            # 3b) Call client.fit() for local DP training and mask computation
            client = FLClient(
                client_model,
                train_loader,
                val_loader,
                test_loader,
                device,
                client_id,
                all_client_ids=list(range(num_clients))
            )
            fit_ins = fl.common.FitIns(
                parameters=ndarrays_to_parameters(get_model_parameters(client_model)),
                config={}
            )
            fit_res = client.fit(fit_ins)

            # Convert Flower Parameters back to NumPy delta
            masked_delta = parameters_to_ndarrays(fit_res.parameters)
            # Accumulate the masked deltas
            agg_masked = [a + m for a, m in zip(agg_masked, masked_delta)]
            # Save metrics for logging
            round_results.append((fit_res.num_examples, fit_res.metrics))

        # 4) Compute average of true deltas (masks cancel out)
        true_sum_delta = agg_masked
        avg_delta = [d / num_clients for d in true_sum_delta]
        # Add average delta to base parameters to get new global params
        new_params = [bp + d for bp, d in zip(base_params, avg_delta)]

        # 5) Log aggregated metrics and update the global model
        metrics = aggregate_metrics(round_results)
        print(f"\n✅ Round {rnd+1} complete. Aggregated accuracy: {metrics['accuracy']:.4f}")
        print(f"📉 Aggregated loss: {metrics['loss']:.4f}")
        wandb.log({
            "round": rnd + 1,
            "Aggregated accuracy": metrics["accuracy"],
            "Aggregated Loss": metrics["loss"]
        })

        # Apply new parameters to the shared global model
        set_model_parameters(global_model, new_params)

    # 6) Return the final global model and parameter list after all rounds
    return global_model, new_params
