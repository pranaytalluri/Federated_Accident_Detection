"""
utils.py

Utility functions for reproducibility, device selection, video loading & preprocessing,
data partitioning, and model parameter serialization/deserialization.

Includes:
  - set_seed: Seed RNGs for reproducibility
  - select_device: Auto-select compute device (MPS/CUDA/CPU) with deterministic settings
  - load_and_sample: Load video frames via PyAV, sample or pad to fixed length
  - video_transform: Apply spatial & temporal augmentations, resize, crop, normalize
  - partition_clients: Split dataset into K balanced clients with train/val/test sets
  - gather_video_paths_and_labels: Scan directories for video files and assign labels
  - get_model_parameters / set_model_parameters: Convert between PyTorch model and NumPy arrays
"""

# === Standard library imports ===
import os                 # File path manipulations
import glob               # File pattern matching
import random             # Python random utilities
import warnings           # To suppress non-critical warnings

# === Third-party imports ===
import av                 # PyAV for video decoding
import numpy as np        # Numerical operations on frames and arrays
import torch               # Core PyTorch for tensors
import torch.nn.functional as F  # Functional API (interpolation, etc.)
from sklearn.model_selection import train_test_split  # Data splitting
from typing import List, Tuple  # Type annotations for clarity
import torchvision.transforms.functional as TF  # Image transformations

# Suppress warnings globally to keep logs concise
warnings.filterwarnings("ignore")


def set_seed(seed: int):
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int): The seed value to apply.
    """
    random.seed(seed)          # Python random
    np.random.seed(seed)       # NumPy random
    torch.manual_seed(seed)    # PyTorch CPU random
    # torch.cuda.manual_seed_all(seed)  # Uncomment if using CUDA


def select_device() -> torch.device:
    """
    Detect and select the appropriate compute device.

    - If MPS (macOS GPU) is available, enable deterministic algorithms.
    - Otherwise, default to CPU. (CUDA fallback not shown here.)

    Returns:
        torch.device: Selected device ("mps" or "cpu").
    """
    # macOS MPS backend check for deterministic behavior
    if torch.backends.mps.is_available():
        torch.use_deterministic_algorithms(True)
        print("[Setup] MPS backend detected. Using deterministic algorithms.")
    else:
        print("[Setup] MPS not available; using CPU.")

    # Create device object
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Setup] Device set to: {device}")
    return device


def load_and_sample(path: str, num_frames: int = 150) -> torch.Tensor:
    """
    Load a video file, decode frames, and uniformly sample or pad to `num_frames`.

    Args:
        path (str): File path to the video.
        num_frames (int): Desired fixed number of frames.

    Returns:
        torch.Tensor: Float tensor of shape [T, H, W, C] with values in [0,1].
    """
    # Open container using PyAV
    container = av.open(path)
    # Decode all frames into RGB numpy arrays
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if not frames:
        raise RuntimeError(f"No frames in video: {path}")

    T = len(frames)
    if T >= num_frames:
        # Uniformly sample indices when there are enough frames
        idxs = np.linspace(0, T - 1, num_frames, dtype=int)
        sampled = [frames[i] for i in idxs]
    else:
        # Pad with last frame if too few frames
        pad_count = num_frames - T
        sampled = frames + [frames[-1]] * pad_count

    # Stack into shape [T, H, W, C]
    video_np = np.stack(sampled, axis=0)
    # Convert to PyTorch tensor and normalize to [0,1]
    tensor = torch.from_numpy(video_np).float() / 255.0
    return tensor


def video_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Apply spatial and temporal augmentations and normalize.

    Steps:
      1. Permute [T,H,W,C] -> [C,T,H,W]
      2. Convert to [T,C,H,W] for frame-by-frame transforms
      3. Random horizontal flip, rotation, color jitter
      4. Resize frames to 128x128 and random crop to 112x112
      5. Permute back to [C,T,H,W]
      6. Normalize with Kinetics mean/std

    Args:
        x (torch.Tensor): Input [T, H, W, C] float tensor.

    Returns:
        torch.Tensor: Augmented and normalized [C, T, H, W] tensor.
    """
    # Kinetics dataset mean & std for normalization
    kinetics_mean = [0.43216, 0.394666, 0.37645]
    kinetics_std  = [0.22803, 0.22145, 0.216989]

    # [T,H,W,C] -> [C,T,H,W]
    x = x.permute(3, 0, 1, 2)
    # [C,T,H,W] -> [T,C,H,W] for frame transforms
    v = x.permute(1, 0, 2, 3)

    # Random horizontal flip (50% chance)
    if random.random() < 0.5:
        v = torch.flip(v, dims=[3])  # Flip width axis

    # Slight random rotation for all frames
    angle = random.uniform(-2, 2)
    v = torch.stack([TF.rotate(frame, angle) for frame in v])

    # Mild color jitter per frame
    brightness = 0.1
    contrast = 0.1
    saturation = 0.1
    hue = 0.01
    v = torch.stack([
        TF.adjust_brightness(frame, 1 + random.uniform(-brightness, brightness))
        for frame in v
    ])
    v = torch.stack([
        TF.adjust_contrast(frame, 1 + random.uniform(-contrast, contrast))
        for frame in v
    ])
    v = torch.stack([
        TF.adjust_saturation(frame, 1 + random.uniform(-saturation, saturation))
        for frame in v
    ])
    v = torch.stack([
        TF.adjust_hue(frame, random.uniform(-hue, hue))
        for frame in v
    ])

    # Resize frames to 128x128
    v = F.interpolate(v, size=(128, 128), mode='bilinear', align_corners=False)

    # Random crop to 112x112
    T, C, H, W = v.shape
    top = random.randint(0, H - 112)
    left = random.randint(0, W - 112)
    v = v[:, :, top:top+112, left:left+112]

    # Permute back to [C,T,H,W]
    v = v.permute(1, 0, 2, 3)

    # Normalize per channel
    mean = torch.tensor(kinetics_mean, device=v.device).view(-1,1,1,1)
    std  = torch.tensor(kinetics_std, device=v.device).view(-1,1,1,1)
    v = (v - mean) / std

    return v


def partition_clients(
    files: List[str],
    labels: List[int],
    K: int = 3,
    val_size: float = 0.3,
    seed: int = 42
) -> List[Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]]:
    """
    Partition dataset into K clients with balanced class distribution.

    For each client:
      1) Stratified split into train & temp (val+test)
      2) Further split temp into val & test

    Args:
        files: List of file paths.
        labels: Corresponding class labels.
        K: Number of client partitions.
        val_size: Fraction of data per client reserved for val+test.
        seed: Random seed for reproducibility.

    Returns:
        List of tuples (train_f, train_l, val_f, val_l, test_f, test_l).
    """
    np.random.seed(seed)

    # Separate by class
    accident_files = [f for f, l in zip(files, labels) if l == 1]
    no_accident_files = [f for f, l in zip(files, labels) if l == 0]

    # Shuffle each class list
    np.random.shuffle(accident_files)
    np.random.shuffle(no_accident_files)

    # Split into K roughly equal chunks per class
    accident_splits = np.array_split(accident_files, K)
    no_accident_splits = np.array_split(no_accident_files, K)

    client_splits = []
    for i in range(K):
        # Combine class chunks for client i
        client_files = list(accident_splits[i]) + list(no_accident_splits[i])
        client_labels = [1]*len(accident_splits[i]) + [0]*len(no_accident_splits[i])
        # Shuffle combined list
        combined = list(zip(client_files, client_labels))
        np.random.shuffle(combined)
        client_files, client_labels = zip(*combined)
        client_files, client_labels = list(client_files), list(client_labels)

        # Stratified train vs temp (val+test)
        train_f, temp_f, train_l, temp_l = train_test_split(
            client_files,
            client_labels,
            test_size=val_size,
            stratify=client_labels,
            random_state=seed + i
        )
        # Split temp into val & test equally
        val_f, test_f, val_l, test_l = train_test_split(
            temp_f,
            temp_l,
            test_size=0.5,
            stratify=temp_l,
            random_state=seed + i
        )

        client_splits.append((train_f, train_l, val_f, val_l, test_f, test_l))

    # Print summary for verification
    print("\n[Partition Summary]")
    for idx, (train_f, train_l, val_f, val_l, test_f, test_l) in enumerate(client_splits):
        print(
            f"Client {idx}: "
            f"Train[{len(train_f)}], Val[{len(val_f)}], Test[{len(test_f)}]"
        )

    return client_splits


def gather_video_paths_and_labels(
    base_dir: str,
    accident_subfolder: str = "accident",
    no_accident_subfolder: str = "no_accident",
    ext: str = "*.mp4",
) -> Tuple[List[str], List[int]]:
    """
    Scan directory for .mp4 videos and assign binary labels.

    Args:
        base_dir: Root directory containing class subfolders.
        accident_subfolder: Name of positive class subfolder.
        no_accident_subfolder: Name of negative class subfolder.
        ext: File extension pattern to search for.

    Returns:
        all_files: Combined list of video paths.
        all_labels: Corresponding list of labels (1 for accident, 0 otherwise).
    """
    # Construct full paths to subfolders
    accident_dir = os.path.join(base_dir, accident_subfolder)
    no_accident_dir = os.path.join(base_dir, no_accident_subfolder)
    print(f"accident_dir: {accident_dir}")
    print(f"no_accident_dir: {no_accident_dir}")

    # Glob all matching video files
    accident_files = glob.glob(os.path.join(accident_dir, ext))
    no_accident_files = glob.glob(os.path.join(no_accident_dir, ext))

    # Combine and label
    all_files = accident_files + no_accident_files
    all_labels = [1]*len(accident_files) + [0]*len(no_accident_files)

    # Print dataset stats
    print(
        f"[Data] Total clips: {len(all_files)} "
        f"(accident={len(accident_files)}, no_accident={len(no_accident_files)})"
    )
    return all_files, all_labels


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as a list of NumPy arrays.

    Args:
        model: PyTorch nn.Module

    Returns:
        List of ndarrays representing each parameter tensor.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: torch.nn.Module, params: List[np.ndarray]):
    """
    Load a list of NumPy arrays into a model's state_dict.

    Args:
        model: PyTorch nn.Module to update.
        params: List of ndarrays corresponding to model parameters.
    """
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(state_dict, strict=True)
