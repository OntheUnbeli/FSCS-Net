import torch.nn as nn
import torch
import torch.nn.functional as F
import collections
from collections import OrderedDict
from Backbone.SegFormer.mix_transformer import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from toolbox.models.AsymFormer.src.convnext import convnext_tiny
from toolbox.model.Mine_New.Block1.RDSM import RDSM
from toolbox.model.Mine_New.Block1.S2 import S2Attention
from toolbox.model.Mine_New.Block1.GMM import GMM
from toolbox.model.Mine_New.Block.Sober import SobelOperator

from toolbox.model.Mine.Block.MLPDecoder import DecoderHead
from toolbox.model.Mine_New.Block.RDMFv1 import RDMF
from toolbox.model.Mine_New.Block.Attention_v1 import Mute
import timm


# import math
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


class perviseHead(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(perviseHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channel, n_classes, 1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.conv(x)

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, s=1, p=0, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class pp_upsample(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1),
            nn.BatchNorm2d(outc),
            nn.PReLU()
        )
    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)

class EnDecoderModel(nn.Module):
    def __init__(self, n_classes=8, backbone='convnext_tiny'):
        super(EnDecoderModel, self).__init__()
        if backbone == 'convnext_tiny':
            self.backboner = convnext_tiny(pretrained=True, drop_path_rate=0.3)
            self.backboned = convnext_tiny(pretrained=True, drop_path_rate=0.3)

        # [64, 128, 320, 640]
        # [64, 128, 320, 512] Segformer
        # [64, 128, 256, 512]
        # [128, 256, 512, 1024] convnext_base
        # [96, 192, 384, 768]

        self.f4_p = perviseHead(768, n_classes)
        self.f3_p = perviseHead(384, n_classes)
        self.f2_p = perviseHead(192, n_classes)
        self.f1_p = perviseHead(96, n_classes)
        self.RDSM4 = RDSM(768)
        self.RDSM3 = RDSM(384)
        self.RDSM2 = RDSM(192)
        self.RDSM1 = RDSM(96)

        self.GMM4 = GMM(768, 15, 20)
        self.GMM3 = GMM(384, 30, 40)
        self.GMM2 = GMM(192, 60, 80)
        self.GMM1 = GMM(96, 120, 160)

        self.conv1 = BasicConv2d(1536, 768, 1)
        self.conv2 = BasicConv2d(768, 384, 1)
        self.conv3 = BasicConv2d(384, 192, 1)
        self.conv4 = BasicConv2d(192, 96, 1)

        self.sober = SobelOperator()

        self.mlpdecoder = DecoderHead(in_channels=[96, 192, 384, 768], num_classes=8)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, rgb, dep):
        features_rgb = self.backboner(rgb)
        features_dep = self.backboned(dep)
        features_rlist = features_rgb
        features_dlist = features_dep

        rf1 = features_rlist[0]
        rf2 = features_rlist[1]
        rf3 = features_rlist[2]
        rf4 = features_rlist[3]

        df1 = features_dlist[0]
        df2 = features_dlist[1]
        df3 = features_dlist[2]
        df4 = features_dlist[3]

        #############################################
        fuse1 = self.RDSM1(rf1, df1)
        fuse2 = self.RDSM2(rf2, df2)
        fuse3 = self.RDSM3(rf3, df3)
        fuse4 = self.RDSM4(rf4, df4)
        # fuse1 = torch.cat([rf1, df1], dim=1)
        # fuse1 = self.conv4(fuse1)
        # fuse2 = torch.cat([rf2, df2], dim=1)
        # fuse2 = self.conv3(fuse2)
        # fuse3 = torch.cat([rf3, df3], dim=1)
        # fuse3 = self.conv2(fuse3)
        # fuse4 = torch.cat([rf4, df4], dim=1)
        # fuse4 = self.conv1(fuse4)



        # f1 = self.GMM1(fuse1)
        # f2 = self.GMM2(fuse2)
        # f3 = self.GMM3(fuse3)
        # f4 = self.GMM4(fuse4)
        # sober_f1 = self.sober(f1)
        # sober_f2 = self.sober(f2)
        # sober_f3 = self.sober(f3)
        # sober_f4 = self.sober(f4)
        # sup4 = self.f4_p(sober_f4)
        # sup3 = self.f3_p(sober_f3)
        # sup2 = self.f2_p(sober_f2)
        # sup1 = self.f1_p(sober_f1)
        # FD_pervise = []
        # FD_pervise.append(sup4)
        # FD_pervise.append(sup3)
        # FD_pervise.append(sup2)
        # FD_pervise.append(sup1)
        sup4 = self.f4_p(fuse4)
        sup3 = self.f3_p(fuse3)
        sup2 = self.f2_p(fuse2)
        sup1 = self.f1_p(fuse1)
        sober_f1 = self.sober(sup1)
        sober_f2 = self.sober(sup2)
        sober_f3 = self.sober(sup3)
        sober_f4 = self.sober(sup4)
        FD_pervise = []
        FD_pervise.append(sup4)
        FD_pervise.append(sup3)
        FD_pervise.append(sup2)
        FD_pervise.append(sup1)


        list = []
        list.append(fuse1)
        list.append(fuse2)
        list.append(fuse3)
        list.append(fuse4)
        out = self.mlpdecoder(list)
        out = self.upsample4(out)

        return out, FD_pervise


    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.backboner.state_dict()
        state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.backboner.load_state_dict(model_dict_r)
        print(f"RGB Loading pre_model ${pre_model}")

        save_model = torch.load(pre_model)
        model_dict_d = self.backboned.state_dict()
        state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_d.keys()}
        model_dict_d.update(state_dict_d)
        self.backboned.load_state_dict(model_dict_d)
        print(f"Depth Loading pre_model ${pre_model}")


if __name__ == '__main__':
    import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device('cuda')
    # rgb = torch.randn(2, 3, 480, 640).to(device)
    # dep = torch.randn(2, 3, 480, 640).to(device)
    # model = EnDecoderModel(backbone='segb2').to(device)
    rgb = torch.randn(2, 3, 480, 640).cuda()
    dep = torch.randn(2, 3, 480, 640).cuda()
    model = EnDecoderModel(n_classes=8, backbone='convnext_tiny').cuda()
    out = model(rgb, dep)
    # for i in range(4):
    #     print(out[i].shape)
    print('out[1]输出结果：', out[1][0].shape)
    # for i in out[1]:
    #     print('out[1]输出结果：', i.shape)
