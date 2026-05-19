import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES_GTSRB = 43


class GTSRB_CNN(nn.Module):
    """
    A CNN adapted for the GTSRB dataset (43 classes, 48x48 input).
    Implements standard CNN components with adjusted layer dimensions for GTSRB.
    """

    def __init__(self, num_classes=NUM_CLASSES_GTSRB):
        """
        Initializes the CNN layers for GTSRB.

        Args:
            num_classes (int): Number of output classes (default: NUM_CLASSES_GTSRB).
        """
        super(GTSRB_CNN, self).__init__()
        # Conv Layer 1: Input 3 channels (RGB), Output 32 filters, Kernel 3x3, Padding 1
        # Processes 48x48 input
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Output shape: (Batch Size, 32, 48, 48)

        # Conv Layer 2: Input 32 channels, Output 64 filters, Kernel 3x3, Padding 1
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        # Output shape: (Batch Size, 64, 48, 48)

        # Max Pooling 1: Kernel 2x2, Stride 2. Reduces spatial dimensions by half.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape: (Batch Size, 64, 24, 24)

        # Conv Layer 3: Input 64 channels, Output 128 filters, Kernel 3x3, Padding 1
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        # Output shape: (Batch Size, 128, 24, 24)

        # Max Pooling 2: Kernel 2x2, Stride 2. Reduces spatial dimensions by half again.
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape: (Batch Size, 128, 12, 12)

        # Calculate flattened feature size after pooling layers
        # This is needed for the input size of the first fully connected layer
        self._feature_size = 128 * 12 * 12  # 18432

        # Fully Connected Layer 1 (Hidden): Maps flattened features to 512 hidden units.
        # Input size MUST match self._feature_size
        self.fc1 = nn.Linear(self._feature_size, 512)
        # Implements Y1 = f(W1 * X_flat + b1), where f is ReLU

        # Fully Connected Layer 2 (Output): Maps hidden units to class logits.
        # Output size MUST match num_classes
        self.fc2 = nn.Linear(512, num_classes)
        # Implements Y_logits = W2 * Y1 + b2

        # Dropout layer for regularization (p=0.5 means 50% probability of dropping a unit)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Defines the forward pass sequence for input tensor x.

        Args:
            x (torch.Tensor): Input batch of images
                                (Batch Size x 3 x IMG_SIZE x IMG_SIZE).

        Returns:
            torch.Tensor: Output logits for each class
                                (Batch Size x num_classes).
        """
        # Apply first Conv block: Conv1 -> ReLU -> Conv2 -> ReLU -> Pool1
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        # Apply second Conv block: Conv3 -> ReLU -> Pool2
        x = self.pool2(F.relu(self.conv3(x)))

        # Flatten the feature map output from the convolutional blocks
        x = x.view(-1, self._feature_size)  # Reshape to (Batch Size, _feature_size)

        # Apply Dropout before the first FC layer (common practice)
        x = self.dropout(x)
        # Apply first FC layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply Dropout again before the output layer
        x = self.dropout(x)
        # Apply the final FC layer to get logits
        x = self.fc2(x)
        return x
