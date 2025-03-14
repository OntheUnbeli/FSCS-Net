import torch
from thop import profile
import torch.nn as nn
from backbone.Shunted.SSA import shunted_b
from networks.Mine.Block.Mute_v5_attention_v1 import Mute
from networks.Mine.Block.HTLF import HTLF
from torch.nn import functional as F
import math

class FakeNet(nn.Module):
    def __init__(self):
        super(FakeNet, self).__init__()

        # Backbone model
        self.backbone_R1 = shunted_b()
        self.backbone_R2 = shunted_b()


        self.conv128_64 = nn.Conv2d(128, 64, 1)
        self.conv512_320 = nn.Conv2d(512, 256, 1)
        self.conv320_128 = nn.Conv2d(256, 128, 1)
        self.conv64_1 = nn.Conv2d(64, 6, 1)

        # [64, 128, 320, 640]
        # [64, 128, 320, 512]
        # [64, 128, 256, 512]
        # Upsample
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Fuse enhance


        self.att4 = Mute(512)
        self.att3 = Mute(320)
        self.att2 = Mute(128)
        self.att1 = Mute(64)

        self.deco3 = HTLF(512, 320)
        self.deco2 = HTLF(320, 128)
        self.deco1 = HTLF(128, 64)



    def forward(self, image1, image2):
        # RGB1
        x1, x2, x3, x4 = self.backbone_R1(image1)

        # RGB2
        i1, i2, i3, i4 = self.backbone_R2(image2)

        r1 = torch.abs(torch.sub(x1, i1))
        r2 = torch.abs(torch.sub(x2, i2))
        r3 = torch.abs(torch.sub(x3, i3))
        r4 = torch.abs(torch.sub(x4, i4))


        fuse_att4 = self.att4(r4)
        fuse_att3 = self.att3(r3)
        fuse_att2 = r2
        fuse_att1 = r1



        # decode
        mult_3 = self.deco3(fuse_att4, fuse_att3)
        mult_2 = self.deco2(fuse_att3, fuse_att2)
        mult_1 = self.deco1(fuse_att2, fuse_att1)

        mul_2 = self.deco2(mult_3, mult_2)
        mul_1 = self.deco1(mult_2, mult_1)

        final_l = self.deco1(mul_2, mul_1)

        final = self.upsample4(final_l)
        final = self.conv64_1(final)

        return final, fuse_att4, fuse_att3, fuse_att2, fuse_att1,\
               mult_3, mult_2, mult_1, mul_2, mul_1


    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.backbone_R1.state_dict()
        state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.backbone_R1.load_state_dict(model_dict_r)
        print(f"RGB Loading pre_model ${pre_model}")

        save_model = torch.load(pre_model)
        model_dict_d = self.backbone_R2.state_dict()
        state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_d.keys()}
        model_dict_d.update(state_dict_d)
        self.backbone_R2.load_state_dict(model_dict_d)
        print(f"Depth Loading pre_model ${pre_model}")



if __name__ == '__main__':

    a = torch.randn(1, 3, 640, 480).cuda()
    b = torch.randn(1, 1, 640, 480).cuda()
    model = FakeNet()
    # model.load_pre('/media/wby/shuju/OR/Remote_Sensing/toolbox/Backbone_Pretrain/ckpt_B.pth')

    model.cuda()
    out = model(a, b)
    print(out[0].shape)
    flops, params = profile(model, inputs=(a, b))
    print('Flops', flops / 1e9, 'G')
    print('Params: ', params / 1e6, 'M')
    # out = model(a, b)
    # print(out)
    # print("out shape", out.shape)