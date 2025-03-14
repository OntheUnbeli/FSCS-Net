import torch.nn as nn
import torch
import torch.nn.functional as F
import collections
from collections import OrderedDict
from Backbone.SegFormer.mix_transformer import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from toolbox.models.AsymFormer.src.convnext import convnext_tiny
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


class EnDecoderModel(nn.Module):
    def __init__(self, n_classes=8, backbone='segb2'):
        super(EnDecoderModel, self).__init__()
        if backbone == 'segb2':
            self.backboner = convnext_tiny(pretrained=True, drop_path_rate=0.3)
            self.backboned = convnext_tiny(pretrained=True, drop_path_rate=0.3)

        # [64, 128, 320, 640]
        # [64, 128, 320, 512] Segformer
        # [64, 128, 256, 512]
        # [128, 256, 512, 1024] convnext_base
        # [96, 192, 384, 768]
        self.rd1 = RDMF(96)
        self.rd2 = RDMF(192)
        self.rd3 = RDMF(384)
        self.rd4 = RDMF(768)

        self.att4 = Mute(768)
        self.att3 = Mute(384)
        self.att2 = Mute(192)
        self.att1 = Mute(96)

        self.f4_p = perviseHead(384, n_classes)
        self.f3_p = perviseHead(192, n_classes)
        self.f2_p = perviseHead(96, n_classes)

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
        fuse1 = self.rd1(rf1, df1)
        fuse2 = self.rd2(rf2, df2)
        fuse3 = self.rd3(rf3, df3)
        fuse4 = self.rd4(rf4, df4)

        fuse4_1 = self.att4(fuse4)  # 512
        fuse3_1 = self.att3(fuse3)  # 320
        # fuse2_1 = self.att2(fuse2)
        # fuse1_1 = self.att1(fuse1)
        # sup1 = self.f4_p(fuse3_1)
        # sup2 = self.f3_p(fuse2)
        # sup3 = self.f2_p(fuse1)
        #
        # FD_pervise = []
        # FD_pervise.append(sup1)
        # FD_pervise.append(sup2)
        # FD_pervise.append(sup3)
        list = []
        list.append(fuse1)
        list.append(fuse2)
        list.append(fuse3_1)
        list.append(fuse4_1)

        out = self.mlpdecoder(list)
        out = self.upsample4(out)
        # print(out.shape)

        return out, list


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
    rgb = torch.randn(2, 3, 480, 640)
    dep = torch.randn(2, 3, 480, 640)
    model = EnDecoderModel(backbone='segb2')
    out = model(rgb, dep)
    # for i in range(4):
    #     print(out[i].shape)
    print('out[1]输出结果：', out[0].shape)
    # for i in out[1]:
    #     print('out[1]输出结果：', i.shape)
