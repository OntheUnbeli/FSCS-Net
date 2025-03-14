import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def apply_frequency_filter(feature_map, filter_type='low_pass', cutoff_freq=0.05):
    # 遍历特征图的每个通道
    filtered_feature_map = torch.zeros_like(feature_map)
    for i in range(feature_map.shape[1]):  # 遍历通道数
        channel = feature_map[:, i, :, :]  # 获取当前通道的特征图

        # 将张量转换为 NumPy 数组，并转换为灰度图像
        channel_numpy = channel.detach().cpu().numpy()[0]
        channel_numpy = np.abs(channel_numpy)  # 取绝对值
        channel_numpy = np.uint8(255 * (channel_numpy - np.min(channel_numpy)) / np.ptp(channel_numpy))

        # 傅里叶变换
        f_transform = np.fft.fft2(channel_numpy)
        f_shift = np.fft.fftshift(f_transform)

        rows, cols = channel_numpy.shape
        crow, ccol = rows // 2, cols // 2

        # 生成频域滤波器
        mask = np.zeros((rows, cols), np.uint8)
        if filter_type == 'high_pass':
            mask[crow - int(cutoff_freq * crow):crow + int(cutoff_freq * crow),
            ccol - int(cutoff_freq * ccol):ccol + int(cutoff_freq * ccol)] = 1
        elif filter_type == 'low_pass':
            mask[crow - int(cutoff_freq * crow):crow + int(cutoff_freq * crow),
            ccol - int(cutoff_freq * ccol):ccol + int(cutoff_freq * ccol)] = 0
            mask = 1 - mask

        # 应用滤波器
        f_shift = f_shift * mask

        # 逆傅里叶变换
        f_ishift = np.fft.ifftshift(f_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # 将 NumPy 数组转换回张量，并复制到结果特征图中的对应通道
        filtered_feature_map[:, i, :, :] = torch.from_numpy(img_back).unsqueeze(0)

    return filtered_feature_map

class RDMF(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super(RDMF, self).__init__()
        self.dim = dim
        self.max_pool_y = nn.AdaptiveMaxPool2d((1, None))
        self.max_pool_x = nn.AdaptiveMaxPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))
        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, bias=False)
        # self.conv1 = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.soft = nn.Softmax(dim=1)
        self.conv1 = BasicConv2d(self.dim, self.dim, kernel_size=(5, 1), stride=1, padding=(2, 0))
        self.conv2 = BasicConv2d(self.dim, self.dim, kernel_size=(1, 5), stride=1, padding=(0, 2))
        self.conv3 = BasicConv2d(self.dim, self.dim, 5, padding=2)
        self.complex_weights = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, r, d):
        add = r + d
        # print(rgbd.shape)
        att1 = self.max_pool_y(add)
        att2 = self.max_pool_x(add)
        att3 = self.avg_pool_y(add)
        att4 = self.avg_pool_x(add)
        attx = att1 + att3
        atty = att2 + att4
        mix = self.conv(attx * atty)
        mix_s = self.soft(mix)
        mix_f = mix_s * add + add

        en1 = self.conv1(mix_f)
        en2 = self.conv2(mix_f)
        en3 = self.conv3(mix_f)
        out = en1 + en2 + en3

        return out

if __name__ == '__main__':
    a = torch.randn(5, 512, 15, 20)
    b = torch.randn(5, 512, 15, 20)
    out = RDMF(512)
    output = out(a, b)
    print(output.shape)