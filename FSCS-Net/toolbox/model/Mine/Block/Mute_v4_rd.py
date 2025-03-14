import torch
from torch import nn
import torch.nn.functional as F

class RDF(nn.Module):
    def __init__(self, dim):
        super(RDF, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False)
        self.conv3 = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1)

    def forward(self, input_rgb, input_depth):
        rgbd = torch.cat((input_rgb, input_depth), dim=1)
        feature_GMP = self.max_pool(rgbd)
        feature_GAP = self.avg_pool(rgbd)

        feature_info = feature_GAP + feature_GMP
        feature_info = self.conv3(feature_info)

        x, _ = torch.max(input_depth, dim=1, keepdim=True)
        x = self.conv2(self.conv1(x))
        mask = torch.sigmoid(x)

        depth_enhance = feature_info * mask + input_rgb

        return depth_enhance

if __name__ == '__main__':
    a = torch.randn(5, 512, 8, 8)
    b = torch.randn(5, 512, 8, 8)
    out = RDF(512)
    output = out(a, b)
    print(output.shape)