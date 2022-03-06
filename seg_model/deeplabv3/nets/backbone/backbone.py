import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from seg_model.deeplabv3.nets.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from seg_model.deeplabv3.nets.backbone import resnet, xception, drn, mobilenet


def build_backbone(backbone, output_stride, BatchNorm, pretrained):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, pretrained)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm, pretrained)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, pretrained)
    else:
        raise NotImplementedError