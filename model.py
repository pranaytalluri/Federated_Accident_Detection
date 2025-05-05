"""
model.py

Defines 3D convolutional neural network architectures and utility functions for
patching BatchNorm layers and pooling layers to ensure compatibility and
performance on various input sizes and devices.

Includes:
  - Small3DCNN: A lightweight 3D CNN for binary classification of video clips
  - SafeInstanceNorm3d: A robust InstanceNorm3d that skips normalization
    when spatial/temporal dimensions are too small
  - replace_batchnorm_with_instancenorm: Recursively replaces BatchNorm3d
    layers with SafeInstanceNorm3d in a given model
  - recursively_patch_pooling_layers: Converts all MaxPool3d and AvgPool3d
    layers into AdaptiveAvgPool3d to avoid hard-coded pooling sizes
  - initialize_model: Factory function to build either an I3D model (with
    patched pooling and normalization) or the Small3DCNN, and move to device
"""

# === Standard library imports ===
import warnings  # To suppress non-critical warnings globally

# === Third-party imports ===
import torch         # Core PyTorch library for tensors and model operations
import torch.nn as nn  # Neural network building blocks (layers, activations, etc.)
from pytorchvideo.models.hub import i3d_r50  # Pretrained I3D-ResNet50 for video

# Suppress all warnings to keep output clean
warnings.filterwarnings("ignore")


class Small3DCNN(nn.Module):
    """
    Lightweight 3D CNN for binary video classification.

    Architecture:
      - 4 convolutional blocks with InstanceNorm3d and ReLU
      - Downsampling by stride=2 in blocks 2-4
      - Global average pooling to produce a [B, 256] feature vector
      - Classifier with one hidden layer and dropout
    """

    def __init__(self, num_classes: int = 2):
        super(Small3DCNN, self).__init__()
        # Stack of convolutional feature extraction layers
        self.features = nn.Sequential(
            # Block 1: preserve spatial dims
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(32, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),

            # Block 2: downsample by half
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(64, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),

            # Block 3: further downsample
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(128, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),

            # Block 4: downsample to small spatial/temporal size
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(256, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),

            # Global pooling to [B, 256, 1, 1, 1]
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        # Fully-connected classifier on pooled features
        self.classifier = nn.Sequential(
            nn.Flatten(),           # Flatten to [B, 256]
            nn.Linear(256, 128),    # Hidden layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),        # Regularization
            nn.Linear(128, num_classes),  # Output logits for classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor and classifier.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 3, T, H, W]
        Returns:
            torch.Tensor: Output logits of shape [B, num_classes]
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class SafeInstanceNorm3d(nn.InstanceNorm3d):
    """
    InstanceNorm3d that bypasses normalization if the spatial/temporal
    dimensions are too small (<=1), avoiding divide-by-zero or nan.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If any spatial or temporal dimension is 1 or less, skip norm
        if input.shape[2] <= 1 and input.shape[3] <= 1 and input.shape[4] <= 1:
            return input
        return super().forward(input)


def replace_batchnorm_with_instancenorm(model: nn.Module) -> nn.Module:
    """
    Recursively traverse `model` and replace all nn.BatchNorm3d
    layers with SafeInstanceNorm3d (affine, no running stats).

    Returns the patched model.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm3d):
            # Replace BatchNorm3d with SafeInstanceNorm3d
            patched = SafeInstanceNorm3d(
                module.num_features,
                affine=True,
                track_running_stats=False,
                eps=1e-3
            )
            setattr(model, name, patched)
        else:
            # Recurse into submodules
            replace_batchnorm_with_instancenorm(module)
    return model


def recursively_patch_pooling_layers(model: nn.Module) -> nn.Module:
    """
    Recursively traverse `model` and convert all MaxPool3d and AvgPool3d
    layers to AdaptiveAvgPool3d((1,1,1)), ensuring global pooling.

    Returns the patched model.
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.MaxPool3d, nn.AvgPool3d)):
            # Replace with adaptive pooling on all dims
            patched = nn.AdaptiveAvgPool3d((1, 1, 1))
            setattr(model, name, patched)
        else:
            recursively_patch_pooling_layers(module)
    return model


def initialize_model(
    num_classes: int = 2,
    model_type: str = "small3dcnn",
    use_mps: bool = None
) -> (nn.Module, torch.device):
    """
    Factory function to build and initialize the chosen model.

    Args:
        num_classes: Number of output classes for classification.
        model_type: 'i3d' for I3D-ResNet50 or 'small3dcnn' for custom model.
        use_mps: If True, force MPS; False force CPU; None auto-select.

    Returns:
        model: Initialized PyTorch model on specified device.
        device: torch.device where the model is placed.
    """
    # Create model architecture
    if model_type == "i3d":
        # Load pretrained I3D ResNet50
        model = i3d_r50(pretrained=True)
        # Patch pooling to global adaptive pooling
        model = recursively_patch_pooling_layers(model)
        # Replace all BatchNorm with InstanceNorm for small inputs
        model = replace_batchnorm_with_instancenorm(model)
        # Modify final projection layer to match num_classes
        in_feats = model.blocks[-1].proj.in_features
        model.blocks[-1].proj = nn.Linear(in_feats, num_classes)
    elif model_type == "small3dcnn":
        # Use the custom lightweight 3D CNN
        model = Small3DCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'i3d' or 'small3dcnn'.")

    # Device selection logic
    if use_mps is None:
        # Auto-select MPS if available, else CPU
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        # Honor explicit user preference if MPS is available
        if use_mps and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Move model to the target device
    model.to(device)
    return model, device
