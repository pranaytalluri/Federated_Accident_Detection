"""
client.py

Client module for federated learning using Flower and Opacus.
Handles:
  - Parameter exchange with the server
  - Local training under Differential Privacy (DP-SGD)
  - Secure aggregation via peer-to-peer mask exchange
  - Local evaluation with alert logging and confusion matrix visualization
"""

# === Standard library imports ===
import json               # For alert log serialization
import traceback          # For printing exception stack traces
import warnings           # To suppress non-critical warnings
import random             # For random seed generation in secure aggregation
from datetime import datetime  # Timestamping for alert logs

# === Third-party imports ===
import torch              # Core PyTorch library for tensors, autograd
import torch.nn as nn     # Neural network layers and loss functions
from torch.optim import Adam  # Optimizer for parameter updates

from opacus import GradSampleModule  # Wrap model to compute per-sample gradients
from opacus.optimizers import DPOptimizer  # DP-wrapped optimizer
from opacus.utils.batch_memory_manager import BatchMemoryManager  # Memory-safe DP batching

import flwr as fl  # Flower federated learning framework
from flwr.common import (
    ndarrays_to_parameters,    # Convert NumPy arrays to Flower Parameters
    parameters_to_ndarrays,    # Convert Flower Parameters to NumPy arrays
    GetParametersIns, GetParametersRes,
    FitIns, FitRes,
    EvaluateIns, EvaluateRes,
    Status, Code               # Status codes for FL communication
)

import matplotlib.pyplot as plt  # For confusion matrix plotting
import numpy as np               # Numerical operations for mask and metrics
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)  # Evaluation metrics
import wandb  # Weights & Biases logging

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")


# --- In-memory stub for peer-to-peer seed exchange ---
# Stores shared random seeds for secure mask generation between client pairs
SEED_STORE = {}


def send_seed_to_peer(my_id: int, peer_id: int, seed: int):
    """
    Store a shared seed for secure aggregation.

    Args:
        my_id: This client's unique identifier.
        peer_id: The other client's ID to pair with.
        seed: Random seed used for mask generation.
    """
    # Use a canonical key tuple (min, max) so both parties index the same entry
    key = (min(my_id, peer_id), max(my_id, peer_id))
    SEED_STORE[key] = seed


def recv_seed_from_peer(my_id: int, peer_id: int) -> int:
    """
    Busy-wait until a peer's seed appears in SEED_STORE, then return it.

    Args:
        my_id: This client's unique identifier.
        peer_id: The other client's ID to pair with.

    Returns:
        The shared random seed from the peer.
    """
    key = (min(my_id, peer_id), max(my_id, peer_id))
    # Spin until the other client writes its seed
    while key not in SEED_STORE:
        pass  # Could add time.sleep() here to reduce CPU usage
    return SEED_STORE[key]


class FLClient(fl.client.Client):
    """
    Flower client implementing:
      - get_parameters: Provide model weights to the server
      - fit: Local training under DP-SGD, mask computation, and return masked delta
      - evaluate: Compute metrics on local test set and log alerts/matrices
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        client_id,
        all_client_ids
    ):
        # Store model and data loaders
        self.model = model
        self.train_loader = train_loader  # DataLoader for training (DP)
        self.val_loader = val_loader      # DataLoader for validation
        self.test_loader = test_loader    # DataLoader for evaluation
        self.device = device              # torch.device (CPU/MPS/CUDA)
        self.client_id = client_id        # Unique FL client ID
        self.all_client_ids = all_client_ids  # List of all client IDs
        # Logging initialization can be enabled if needed
        # print(f"[Client {client_id}] Initialized with model and loaders.")

    def get_parameters(self) -> GetParametersRes:
        """
        Send local model parameters to the FL server.

        Returns:
            GetParametersRes containing status and parameter list.
        """
        print(f"[Client {self.client_id}] get_parameters called.")
        # Convert each tensor in state_dict to NumPy for Flower
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(params)
        )

    def set_parameters(self, parameters: fl.common.Parameters):
        """
        Load new global parameters into the local model.

        Args:
            parameters: Flower Parameters containing serialized weights.
        """
        ndarrays = parameters_to_ndarrays(parameters)
        # Map model's state_dict keys to incoming arrays
        param_dict = zip(self.model.state_dict().keys(), ndarrays)
        state_dict = {k: torch.tensor(v) for k, v in param_dict}
        self.model.load_state_dict(state_dict, strict=True)
        # print(f"[Client {self.client_id}] set_parameters applied.")

    def fit(self, ins: FitIns) -> FitRes:
        """
        Perform one round of local DP training and generate a masked delta.

        Steps:
        1) Load global snapshot into model
        2) Wrap model for per-sample gradients
        3) Run DP-SGD via BatchMemoryManager
        4) Compute validation metrics
        5) Compute raw weight delta and apply pairwise masks
        6) Return masked delta and metrics
        """
        # 1) Load global weights from server
        base_ndarrays = parameters_to_ndarrays(ins.parameters)
        self.set_parameters(ins.parameters)

        # 2) Switch to training mode and enable per-sample gradient tracking
        self.model.train()
        self.model = GradSampleModule(self.model)

        # Loss and optimizer setup
        criterion = nn.CrossEntropyLoss()
        num_local_epochs = 5  # Number of local epochs per round
        base_optimizer = Adam(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        dp_optimizer = DPOptimizer(
            base_optimizer,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            expected_batch_size=self.train_loader.batch_size,
        )

        # 3) Run DP training safely within memory constraints
        with BatchMemoryManager(
             data_loader=self.train_loader,
             max_physical_batch_size=1,
             optimizer=dp_optimizer,
        ) as memory_safe_loader:
            for epoch in range(num_local_epochs):
                epoch_loss, correct, total, batches = 0.0, 0, 0, 0
                # Batch loop
                for videos, labels in memory_safe_loader:
                    # Move data to the correct device
                    videos, labels = (
                        videos.to(self.device), labels.to(self.device)
                    )
                    dp_optimizer.zero_grad()
                    logits = self.model(videos)
                    loss = criterion(logits, labels)
                    # Compute gradients
                    loss.backward()
                    # DP step: add noise + clipping
                    dp_optimizer.step()

                    # Track accuracy
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    epoch_loss += loss.item()
                    batches += 1

                    # Ensure operations finish on GPU (MPS/CUDA)
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    elif torch.cuda.is_available():
                        torch.cuda.synchronize()

                # Compute averaged training metrics
                avg_train_loss = epoch_loss / batches if batches else 0.0
                train_accuracy = correct / total if total else 0.0

                # 4) Validation pass (no gradient)
                self.model.eval()
                val_loss, correct_val, total_val = 0.0, 0, 0
                with torch.no_grad():
                    for videos, labels in self.val_loader:
                        videos, labels = (
                            videos.to(self.device), labels.to(self.device)
                        )
                        logits = self.model(videos)
                        loss_val = criterion(logits, labels)
                        val_loss += loss_val.item()
                        preds_val = logits.argmax(dim=1)
                        correct_val += (preds_val == labels).sum().item()
                        total_val += labels.size(0)
                avg_val_loss = (
                    val_loss / len(self.val_loader)
                    if len(self.val_loader) > 0 else 0.0
                )
                val_accuracy = correct_val / total_val if total_val else 0.0

                # Log metrics to Weights & Biases
                wandb.log({
                    f"client_{self.client_id}/train_loss_epoch{epoch+1}": avg_train_loss,
                    f"client_{self.client_id}/train_accuracy_epoch{epoch+1}": train_accuracy,
                    f"client_{self.client_id}/val_loss_epoch{epoch+1}": avg_val_loss,
                    f"client_{self.client_id}/val_accuracy_epoch{epoch+1}": val_accuracy,
                    "round": epoch + 1
                })
                # Return to train mode for next epoch
                self.model.train()

        # 5) Compute raw model delta (new - old)
        new_ndarrays = [val.cpu().numpy() for val in self.model.state_dict().values()]
        delta = [new - base for new, base in zip(new_ndarrays, base_ndarrays)]

        # 6) Secure aggregation: pairwise mask exchange
        mask = [np.zeros_like(d) for d in delta]
        for peer_id in self.all_client_ids:
            if peer_id == self.client_id:
                continue  # Skip self
            if self.client_id < peer_id:
                # Generate and send seed to peer
                seed = random.getrandbits(32)
                send_seed_to_peer(self.client_id, peer_id, seed)
            else:
                # Receive seed from peer
                seed = recv_seed_from_peer(self.client_id, peer_id)
            rng = np.random.RandomState(seed)
            # Apply symmetric masking noise
            for i, d in enumerate(delta):
                noise = rng.randn(*d.shape)
                mask[i] += noise if self.client_id < peer_id else -noise

        # Add mask to raw delta
        masked_delta = [d + m for d, m in zip(delta, mask)]

        # Return masked update and metrics
        return FitRes(
            status=Status(code=Code.OK, message="Training complete"),
            parameters=ndarrays_to_parameters(masked_delta),
            num_examples=len(self.train_loader.dataset),
            metrics={
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy
            }
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate global model on the local test set:
          - Compute accuracy, precision, recall, F1
          - Log high-confidence alerts to JSON
          - Plot and log confusion matrix to W&B
        """
        print(f"[Client {self.client_id}] Evaluate: Starting evaluation.")
        # Load provided parameters
        self.set_parameters(ins.parameters)
        self.model.eval()
        try:
            # === Test loop ===
            correct_test, total_test = 0, 0
            all_test_preds, all_test_labels = [], []
            with torch.no_grad():
                for videos, labels in self.test_loader:
                    videos, labels = (
                        videos.to(self.device), labels.to(self.device)
                    )
                    preds = self.model(videos).argmax(dim=1)
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    elif torch.cuda.is_available():
                        torch.cuda.synchronize()
                    correct_test += (preds == labels).sum().item()
                    total_test += labels.size(0)
                    all_test_preds.extend(preds.cpu().numpy())
                    all_test_labels.extend(labels.cpu().numpy())

            # === Alert logging: record high-confidence predictions ===
            alert_logs = []
            with torch.no_grad():
                for videos, labels in self.test_loader:
                    videos, labels = (
                        videos.to(self.device), labels.to(self.device)
                    )
                    logits = self.model(videos)
                    probs = torch.softmax(logits, dim=1)
                    max_probs, preds = torch.max(probs, dim=1)
                    for i in range(videos.size(0)):
                        confidence = max_probs[i].item()
                        if confidence >= 0.85:
                            alert_logs.append({
                                "timestamp": datetime.now().isoformat(),
                                "client_id": self.client_id,
                                "true_label": int(labels[i].item()),
                                "predicted_label": int(preds[i].item()),
                                "confidence": confidence
                            })
            # Save alerts if any
            if alert_logs:
                alert_path = f"client_{self.client_id}_alerts.json"
                with open(alert_path, "w") as f:
                    json.dump(alert_logs, f, indent=4)
                print(f"[Alert] Client {self.client_id}: {len(alert_logs)} high-confidence alerts logged → {alert_path}")
            else:
                print(f"[Alert] Client {self.client_id}: No high-confidence alerts.")

            # === Compute summary metrics ===
            test_acc = correct_test / total_test if total_test > 0 else 0.0
            test_precision = precision_score(all_test_labels, all_test_preds, zero_division=0)
            test_recall = recall_score(all_test_labels, all_test_preds, zero_division=0)
            test_f1 = f1_score(all_test_labels, all_test_preds, zero_division=0)

            # === Plot confusion matrix ===
            cm = confusion_matrix(all_test_labels, all_test_preds)
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(cm, interpolation="nearest")
            ax.figure.colorbar(im, ax=ax)
            labels = ["Positive", "Negative"]  # x-axis
            ax.set(
                xticks=np.arange(len(labels)),
                yticks=np.arange(len(labels)),
                xticklabels=labels,
                yticklabels=["True", "False"],
                xlabel="Predicted",
                ylabel="Actual",
                title="Confusion Matrix"
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            fontsize=18, fontweight="bold")
            plt.tight_layout()

            # === Log metrics & plot to W&B ===
            wandb.log({
                f"client_{self.client_id}/test_accuracy": test_acc,
                f"client_{self.client_id}/test_precision": test_precision,
                f"client_{self.client_id}/test_recall": test_recall,
                f"client_{self.client_id}/test_f1": test_f1,
                f"client_{self.client_id}/test_confusion_matrix": wandb.Image(fig)
            })

            return EvaluateRes(
                status=Status(code=Code.OK, message="Evaluation successful"),
                loss=1.0 - test_acc,
                num_examples=total_test,
                metrics={
                    "test_accuracy": test_acc,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_f1": test_f1
                }
            )
        except Exception as e:
            # Print exception details for debugging
            print(f"[Client {self.client_id}] Exception during evaluation: {e}")
            traceback.print_exc()
            raise e
