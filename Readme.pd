# Privacy Preserving Traffic Accident Detection Using Federated Learning

This repository implements a federated learning pipeline for binary video classification (accident vs. no-accident) using Flower and Opacus. It supports two model architectures:

* **Small3DCNN**: A custom lightweight 3D convolutional neural network.
* **I3D-ResNet50**: Pretrained I3D model with patched pooling and normalization.

Differential Privacy (DP-SGD) is applied locally on each client using Opacus, and secure aggregation is performed via pairwise mask exchange.

---

## 📁 Repository Structure

```
Federated_Accident_Detection/
├── client.py                   # Flower client: get/set params, DP training, evaluation
├── data.py                     # CADPClipDataset: loads, augments, and returns video clips
├── evaluate.py                 # Final evaluation script for global model
├── main.py                     # Entrypoint: sets up, runs federated rounds, and evaluates
├── model.py                    # Defines Small3DCNN and I3D patching & initialization
├── train.py                    # run_federated_rounds: orchestrates federated training loop
├── utils.py                    # Helper functions (seeds, device, loading, transforms, partition)
├── requirements.txt            # List of packages versions installed in our environment
└── README.md                   # This file
```

Additional folders:

* `Data/accident` & `Data/no_accident`: video clips for each class.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/pranaytalluri/Federated_Accident_Detection.git
cd Federated_Accident_Detection
```

### 2. Create a Conda environment

```bash
conda create -n fl_env python=3.10
conda activate fl_env
pip install -r requirements.txt
```

**Note:** On macOS, if you plan to use MPS, ensure you have the latest PyTorch build with MPS support.

### 3. Organize your dataset

Place `.mp4` files under:

```
/Data/accident/*.mp4
/Datas/no_accident/*.mp4
```

### 4. Run federated training & evaluation

```bash
python main.py
```

This will:

1. Initialize Weights & Biases for logging.
2. Partition data across 2 clients (train/val/test).
3. Run 3 federated rounds with DP-SGD.
4. Evaluate the final global model on each client's test split.

---

## 🛠️ Configuration

Parameters are configured in `main.py` via WandB:

```python
wandb.init(
    project="FL_I3D_Training",
    name="run_federated_simulation",
    config={
        "rounds": 3,
        "lr": 1e-4
    }
)
```

* `rounds`: Number of federated communication rounds.
* `lr`: Base learning rate for local DP-SGD.

You can modify `config` or pass overrides via the WandB CLI.

---

## 🔍 Code Overview

### `client.py`

* **`get_parameters` / `set_parameters`**: Serialize and load model weights.
* **`fit`**: Performs local DP-SGD with Opacus and BatchMemoryManager, computes masked weight delta.
* **`evaluate`**: Runs evaluation on local test loader, logs metrics & confusion matrix.

### `train.py`

* **`run_federated_rounds`**: Loops over rounds, orchestrates client training, aggregates masked deltas, and updates global model.

### `model.py`

* **`Small3DCNN`**: Custom 3D CNN architecture.
* **I3D patching**: Converts BatchNorm→InstanceNorm, Pooling→AdaptiveAvgPool.

### `data.py` & `utils.py`

* Data loading, augmentation (spatial & temporal), and partitioning utilities.

---

## 📈 Logging & Visualization

* **Weights & Biases**: Metrics for each client and round are logged (loss, accuracy, precision, recall, F1).
* **Confusion Matrix**: Plotted and sent to WandB per evaluation.

Visit your WandB project dashboard to monitor live training and evaluation.

---

## 🔮 Next Steps

* Increase `num_rounds` and `num_local_epochs` for better convergence.
* Experiment with different noise multipliers and clipping norms.
* Add advanced video augmentations (MixUp, CutMix, temporal jitter).
* Scale to more clients or different data distributions.

---

