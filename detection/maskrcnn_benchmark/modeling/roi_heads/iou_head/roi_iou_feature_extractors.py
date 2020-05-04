# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc


@registry.ROI_IOU_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_IOU_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_IOU_HEAD.POOLER_SCALES
        pooler_type = cfg.MODEL.ROI_IOU_HEAD.POOLER_TYPE
        pooler = Pooler(
            pooler_type=pooler_type,
            output_size=(resolution, resolution),
            scales=scales,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_IOU_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_IOU_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


@registry.ROI_IOU_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractorRnd")
class FPN2MLPFeatureExtractorRnd(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractorRnd, self).__init__()

        resolution = cfg.MODEL.ROI_IOU_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_IOU_HEAD.POOLER_SCALES
        pooler_type = cfg.MODEL.ROI_IOU_HEAD.POOLER_TYPE
        pooler = Pooler(
            pooler_type=pooler_type,
            output_size=(resolution, resolution),
            scales=scales,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_IOU_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_IOU_HEAD.USE_GN
        self.pooler = pooler
        self.fc6_iou = make_fc(input_size, representation_size, use_gn)
        self.fc7_iou = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6_iou(x))
        x = F.relu(self.fc7_iou(x))

        return x


def make_roi_iou_feature_extractor(cfg, in_channels):
    func = registry.ROI_IOU_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_IOU_HEAD.FEATURE_EXTRACTOR
    ]

    net = func(cfg, in_channels)

    if not cfg.MODEL.ROI_IOU_HEAD.TRAIN_HEAD:
        for p in net.parameters():
            p.requires_grad = False

    return net
