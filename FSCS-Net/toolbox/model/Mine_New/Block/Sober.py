import torch
import torch.nn.functional as F
from torch import nn

class SobelOperator(nn.Module):
    def __init__(self):
        super(SobelOperator, self).__init__()
        # Sobel filters for x and y directions
        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, feature_map):
        """
        Applies Sobel operator to a feature map.

        Args:
            feature_map (torch.Tensor): Input feature map of shape (N, C, H, W).

        Returns:
            torch.Tensor: Gradient magnitude of the feature map.
        """
        # Ensure the Sobel filters are on the same device as the feature map
        device = feature_map.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)

        # Separate Sobel operation for each channel
        gradients_x, gradients_y = [], []
        for c in range(feature_map.size(1)):
            grad_x = F.conv2d(feature_map[:, c:c+1], self.sobel_x, padding=1)
            grad_y = F.conv2d(feature_map[:, c:c+1], self.sobel_y, padding=1)
            gradients_x.append(grad_x)
            gradients_y.append(grad_y)

        # Stack gradients along the channel dimension
        gradients_x = torch.cat(gradients_x, dim=1)
        gradients_y = torch.cat(gradients_y, dim=1)

        # Compute gradient magnitude
        gradient_magnitude = torch.sqrt(gradients_x ** 2 + gradients_y ** 2)

        return gradient_magnitude

# Example usage
if __name__ == "__main__":
    # Create a random feature map (N, C, H, W)
    feature_map = torch.randn(1, 3, 64, 64)  # Example: Batch of 1, 3 channels, 64x64 size

    sobel = SobelOperator()
    result = sobel(feature_map)

    print("Input feature map shape:", feature_map.shape)
    print("Output gradient magnitude shape:", result.shape)