import torch
from thop import profile
import torch.nn as nn
from Remote_Sensing.toolbox.backbone.VMamba.classification.models.vmamba1 import vmamba_tiny_s1l8
from torch.nn import functional as F

class FakeNet(nn.Module):
    def __init__(self):
        super(FakeNet, self).__init__()

        # Backbone model
        self.backbone_R = vmamba_tiny_s1l8()

        self.conv1 = nn.Conv2d(768, 384, 1)
        self.conv2 = nn.Conv2d(384, 192, 1)
        self.conv3 = nn.Conv2d(192, 96, 1)
        self.conv4 = nn.Conv2d(96, 6, 1)

        # [64, 128, 320, 640]
        # [64, 128, 320, 512]
        # [64, 128, 256, 512]
        # [96, 192, 384, 768]
        # Upsample
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Fuse enhance


    def forward(self, x):
        # RGB
        x1, x2, x3, x4 = self.backbone_R(x)
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        print(x4.shape)



        # decode
        r4 = self.conv1(x4)
        r4 = self.upsample2(r4)
        r3 = r4 + x3
        r3 = self.conv2(r3)
        r3 = self.upsample2(r3)
        r2 = r3 + x2
        r1 = self.conv3(r2)
        r1 = self.upsample2(r1)
        r0 = r1 + x1
        final = self.conv4(r0)
        final = self.upsample4(final)

        return final


    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.backbone_R.state_dict()
        state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.backbone_R.load_state_dict(model_dict_r)
        print(f"RGB Loading pre_model ${pre_model}")
        #
        # save_model = torch.load(pre_model)
        # model_dict_d = self.mit_b5_D.state_dict()
        # state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_d.keys()}
        # model_dict_d.update(state_dict_d)
        # self.mit_b5_D.load_state_dict(model_dict_d)
        # print(f"Depth Loading pre_model ${pre_model}")



if __name__ == '__main__':

    a = torch.randn(5, 3, 224, 224).cuda()
    # b = torch.randn(5, 3, 256, 256).cuda()
    model = FakeNet()
    model.load_pre('/media/wby/shuju/OR/Remote_Sensing/toolbox/Backbone_Pretrain/vssm1_tiny_0230s_ckpt_epoch_264.pth')

    model.cuda()
    out = model(a, )
    print(out.shape)
    flops, params = profile(model, inputs=(a, ))
    print('Flops', flops / 1e9, 'G')
    print('Params: ', params / 1e6, 'M')
    # out = model(a, b)
    # print(out)
    # print("out shape", out.shape)