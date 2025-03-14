from .metrics import averageMeter, runningScore
from .log import get_logger
from .loss import MscCrossEntropyLoss
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, adjust_lr
from .ranger.ranger import Ranger
from .ranger.ranger913A import RangerVA
from .ranger.rangerqh import RangerQH

def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'sunrgbd224', 'SUIM']

    if cfg['dataset'] == 'nyuv2':
        from .datasets.suim import NYUv2
        return NYUv2(cfg, mode='train'), NYUv2(cfg, mode='test')
    if cfg['dataset'] == 'sunrgbd':
        from .datasets.sunrgbd import SUNRGBD
        return SUNRGBD(cfg, mode='train'), SUNRGBD(cfg, mode='test')
    if cfg['dataset'] == 'sunrgbd224':
        from .datasets.sunrgbd224 import SUNRGBD224
        return SUNRGBD224(cfg, mode='train'), SUNRGBD224(cfg, mode='test')
    if cfg['dataset'] == 'SUIM':
        from .datasets.suim import SUIM
        return SUIM(cfg, mode='train'), SUIM(cfg, mode='test')


def get_model(cfg):
    # if cfg['model_name'] == 'DGPI':
    #     from .model.DGPINet_T import EnDecoderModel
    #     return EnDecoderModel(n_classes=8, backbone='segb2')

    if cfg['model_name'] == 'SGFNet':
        from .models.SGFNet.SGFNet import SGFNet
        return SGFNet()
    if cfg['model_name'] == 'fakev1':
        from .model.Mine_New.Net.FakeNetv2 import EnDecoderModel
        return EnDecoderModel(n_classes=8, backbone='segb2')
    if cfg['model_name'] == 'fakev2':
        from .model.Mine_New.Net.FakeNetv2_Convnext import EnDecoderModel
        return EnDecoderModel(n_classes=8, backbone='segb2')
    if cfg['model_name'] == 'FakeNetv2_Shunted':
        from .model.Mine_New.Net.FakeNetv2_Shunted import EnDecoderModel
        return EnDecoderModel(n_classes=8, backbone='segb2')
    if cfg['model_name'] == 'True':
        from .model.Mine_New.Net1.TrueNetv1 import EnDecoderModel
        return EnDecoderModel(n_classes=8, backbone='convnext_tiny')


    if cfg['model_name'] == 'Asym':
        from toolbox.models.AsymFormer.src.AsymFormer import B0_T
        return B0_T(num_classes=8)
    if cfg['model_name'] == 'HYT':
        from toolbox.models.HYT.net1 import model_1
        return model_1()


    # if cfg['model_name'] == 'bbsnet':
    #     from .models.BBSnetmodel.BBSnet import BBSNet
    #     return BBSNet(n_class=cfg['n_classes'])

def get_teacher_model(cfg):
    if cfg['model_name'] == 'tex1':
        from .models.text1_Net.models.text1 import EncoderDecoder
        return EncoderDecoder()

def get_mutual_model(cfg):
    if cfg['model_name'] == 'fakev1':
        from .model.Mine_New.Net1.TrueNetv1 import EnDecoderModel
        return EnDecoderModel(n_classes=8, backbone='convnext_tiny')



