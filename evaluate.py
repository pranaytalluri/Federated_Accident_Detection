"""
evaluate.py

Final evaluation script for the global federated model.
Combines each client's hold-out test set, builds DataLoaders, and
computes final metrics (accuracy, precision, recall, F1) via an FLClient.
"""

# === Standard library imports ===
import warnings  # To suppress non-critical warnings

# === Third-party imports ===
from torch.utils.data import DataLoader  # For batching test data
import flwr as fl  # Flower framework for federated evaluation
from flwr.common import ndarrays_to_parameters  # Convert NumPy arrays to FL Parameters

# === Local imports ===
from client import FLClient  # Custom Flower client with evaluate() method
from data import CADPClipDataset  # Dataset wrapper for CADP video clips

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def evaluate_model(global_model, averaged_params, client_splits, device):
    """
    Evaluate the trained global model on each client's test split.

    Args:
        global_model (torch.nn.Module): The final federated model.
        averaged_params (List[np.ndarray]): Flattened model parameters after aggregation.
        client_splits (List[Tuple]): List of splits per client: (train_f, train_l, val_f, val_l, test_f, test_l).
        device (torch.device): Device on which to run evaluation (CPU/MPS/CUDA).
    """
    # Inform user that federated training has finished and evaluation begins
    print("\n🏁 Federated learning completed. Evaluating global model...\n")

    # Loop over each client's data split
    for client_id, split in enumerate(client_splits):
        # Unpack only the validation and test portions (train is not used here)
        _, _, val_f, val_l, test_f, test_l = split

        # Build a Dataset for this client's test videos
        test_ds = CADPClipDataset(test_f, test_l)
        # DataLoader for testing: batch size 1, no shuffle for deterministic order
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        # Dummy loader to satisfy FLClient signature (train/val loaders unused in evaluate)
        dummy_loader = DataLoader(test_ds, batch_size=1)

        # Instantiate a Flower client for evaluation only
        client = FLClient(
            global_model,
            dummy_loader,    # train_loader unused here
            dummy_loader,    # val_loader unused here
            test_loader,
            device,
            client_id       # Unique client identifier
        )

        # Create an EvaluateIns containing the global parameters for client
        eval_ins = fl.common.EvaluateIns(
            parameters=ndarrays_to_parameters(averaged_params),
            config={}  # No extra config needed
        )

        # Run evaluation and collect metrics
        eval_res = client.evaluate(eval_ins)

        # Print concise summary of final test metrics for this client
        print(
            f"[Client {client_id}] Final Test Acc:   {eval_res.metrics['test_accuracy']:.4f}, "
            f"Precision: {eval_res.metrics['test_precision']:.4f}, "
            f"Recall:    {eval_res.metrics['test_recall']:.4f}, "
            f"F1 Score:  {eval_res.metrics['test_f1']:.4f}"
        )
