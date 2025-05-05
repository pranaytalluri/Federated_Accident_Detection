"""
data.py

Defines the CADPClipDataset class for loading, augmenting,
 and preparing video clips for model training and evaluation.
Wraps low-level video loading and transformation utilities into
a PyTorch Dataset interface.
"""

# === Standard library imports ===
import warnings  # To suppress non-critical warnings

# === Third-party imports ===
from torch.utils.data import Dataset  # Base class for creating PyTorch Datasets

# === Local imports ===
from utils import load_and_sample, video_transform
#   load_and_sample: function to load and uniformly sample/pad video frames
#   video_transform: function to apply spatial & temporal augmentations + normalization

# Suppress all warnings to keep console output clean
warnings.filterwarnings("ignore")


class CADPClipDataset(Dataset):
    """
    PyTorch Dataset for CADP video clips.

    Each sample:
      1) Loads raw frames from disk via load_and_sample (on CPU)
      2) Applies augmentations and normalization via video_transform (on CPU)
      3) Moves the processed tensor to GPU (MPS/CUDA) for training

    Args:
        files (List[str]): List of file paths for each video clip.
        labels (List[int]): Corresponding list of integer class labels.
        num_frames (int): Number of frames to sample or pad per clip.
    """

    def __init__(self, files, labels, num_frames=150):
        # Store file paths and labels for indexing
        self.files = files
        self.labels = labels
        # Fixed number of frames per video (for uniform tensor shape)
        self.num_frames = num_frames

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Fetches and processes the video clip at index idx.

        Steps:
          1. Load and uniformly sample frames from disk (CPU-bound)
          2. Apply spatial/temporal augmentations & normalization (CPU-bound)
          3. Transfer the resulting tensor to GPU memory (MPS/CUDA)

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            video (torch.Tensor): Tensor of shape [C, T, H, W] on GPU.
            label (int): Corresponding class label.
        """
        # Get file path and label for this index
        path = self.files[idx]
        label = self.labels[idx]

        # 1) Load raw video frames and sample/pad to fixed length
        #    Returns a CPU tensor of shape [T, H, W, C] with values in [0,1]
        video = load_and_sample(path, self.num_frames)

        # 2) Apply augmentations & normalization (spatial transforms, jitter, etc.)
        #    Returns a CPU tensor of shape [C, T, H, W]
        video = video_transform(video)

        # 3) Transfer tensor to GPU (MPS/CUDA) after preprocessing
        #    Ensures subsequent model training/inference uses GPU acceleration
        video = video.to("mps")

        return video, label
