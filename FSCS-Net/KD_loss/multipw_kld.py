import torch
import torch.nn as nn
import torch.nn.functional as F
from KD_loss.loss import KLDLoss

class Multipwkld(nn.Module):
    def __init__(self, initial_scale=0.5, initial_alpha=0.5):
        super(Multipwkld, self).__init__()
        # 将 scale 设置为可学习的参数
        self.scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32, requires_grad=True))
        self.maxpool = None  # 延迟初始化
        self.kld = KLDLoss(tau=1)
        self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32))

    def forward(self, preds_S, preds_T):
        # Detach teacher features to avoid gradient flow
        feat_T = preds_T.detach()
        feat_S = preds_S

        # Clamp scale to prevent invalid values
        scale_clamped = torch.clamp(self.scale, min=0.1, max=1.0).item()

        # Dynamically initialize the maxpool layer
        if self.maxpool is None or self.scale.requires_grad:
            total_w, total_h = feat_T.size(2), feat_T.size(3)
            patch_w = max(1, int(total_w * scale_clamped))
            patch_h = max(1, int(total_h * scale_clamped))
            self.maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True)

        # Apply maxpool and compute the loss
        pooled_feat_S = self.maxpool(feat_S)
        pooled_feat_T = self.maxpool(feat_T)
        KLDloss = self.kld(feat_S, feat_T)
        loss = sim_dis_compute(pooled_feat_S, pooled_feat_T)
        alpha = torch.sigmoid(self.alpha)
        loss_t = alpha * KLDloss + (1 - alpha) * loss
        return loss_t

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S)) ** 2).mean()
    return sim_err

def similarity(feat):
    feat_normalized = F.normalize(feat.view(feat.size(0), feat.size(1), -1), p=2, dim=1)
    return torch.einsum('icm,icn->imn', [feat_normalized, feat_normalized])

# Example usage
if __name__ == '__main__':
    # Simulate student and teacher feature maps
    x = torch.randn(2, 3, 480, 640)  # Student features
    y = torch.randn(2, 3, 480, 640)  # Teacher features

    # Initialize the loss criterion with a learnable scale
    criterion = Multipwkld(initial_scale=0.5)
    optimizer = torch.optim.SGD(criterion.parameters(), lr=1e-3)

    for epoch in range(5):
        optimizer.zero_grad()
        loss = criterion(x, y)
        loss.backward()
        optimizer.step()

        # Print the updated scale and loss
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}, Scale = {criterion.scale.item()}")