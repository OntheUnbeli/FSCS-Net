import torch
from thop import profile
import torch.nn as nn
# from COA_RGBD_SOD.Backbone.SegFormer.mix_transformer import mit_b5
from Backbone.VMamba.classification.models.vmamba import Backbone_VSSM
from toolbox.model.Mine.Block.Mute_v5_attention_v1 import Mute
# from COA_RGBD_SOD.al.models.Fake_mine.Block.Mute_v1_1 import DepthWiseConv
from toolbox.model.Mine.Block.Mute_v4_rd import RDF
from toolbox.model.Mine.Block.HTLF import HTLF
from torch.nn import functional as F
import math

class FakeNet(nn.Module):
    def __init__(self):
        super(FakeNet, self).__init__()

        # Backbone model
        self.mit_b5_R = Backbone_VSSM()
        self.mit_b5_D = Backbone_VSSM()


        self.conv128_64 = nn.Conv2d(192, 64, 1)
        self.conv512_320 = nn.Conv2d(768, 256, 1)
        self.conv320_128 = nn.Conv2d(384, 128, 1)
        self.conv64_1 = nn.Conv2d(96, 6, 1)

        # [64, 128, 320, 640]
        # [64, 128, 320, 512]
        # [64, 128, 256, 512]
        # [96, 192, 384, 768]
        # Upsample
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Fuse enhance
        self.rd_fuse_4 = RDF(768)
        self.rd_fuse_3 = RDF(384)
        self.rd_fuse_2 = RDF(192)
        self.rd_fuse_1 = RDF(96)

        self.att4 = Mute(768)
        self.att3 = Mute(384)
        self.att2 = Mute(192)
        self.att1 = Mute(96)

        self.deco3 = HTLF(768, 384)
        self.deco2 = HTLF(384, 192)
        self.deco1 = HTLF(192, 96)



    def forward(self, x, x_depth):
        # RGB
        x1, x2, x3, x4 = self.mit_b5_R(x)

        # Depth
        d1, d2, d3, d4 = self.mit_b5_D(x_depth)

        # FUSE ATT
        fuse1 = self.rd_fuse_1(x1, d1)
        fuse2 = self.rd_fuse_2(x2, d2)
        fuse3 = self.rd_fuse_3(x3, d3)
        fuse4 = self.rd_fuse_4(x4, d4)

        fuse_att4 = self.att4(fuse4)
        fuse_att3 = fuse3
        fuse_att2 = fuse2
        fuse_att1 = fuse1



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
        model_dict_r = self.mit_b5_R.state_dict()
        state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.mit_b5_R.load_state_dict(model_dict_r)
        print(f"RGB Loading pre_model ${pre_model}")

        save_model = torch.load(pre_model)
        model_dict_d = self.mit_b5_D.state_dict()
        state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_d.keys()}
        model_dict_d.update(state_dict_d)
        self.mit_b5_D.load_state_dict(model_dict_d)
        print(f"Depth Loading pre_model ${pre_model}")



if __name__ == '__main__':

    a = torch.randn(5, 3, 256, 256).cuda()
    b = torch.randn(5, 3, 256, 256).cuda()
    model = FakeNet()
    model.load_pre('/media/wby/shuju/OR/Remote_Sensing/toolbox/Backbone_Pretrain/vssm1_tiny_0230s_ckpt_epoch_264.pth')

    model.cuda()
    out = model(a, b)
    print(out[0].shape)
    flops, params = profile(model, inputs=(a, b))
    print('Flops', flops / 1e9, 'G')
    print('Params: ', params / 1e6, 'M')
    # out = model(a, b)
    # print(out)
    # print("out shape", out.shape)