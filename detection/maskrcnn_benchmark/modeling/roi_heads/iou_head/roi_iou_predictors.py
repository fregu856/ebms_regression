# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn


@registry.ROI_IOU_PREDICTOR.register("IoUNetPredictor")
class IoUNetPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(IoUNetPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        num_iou_pred_classes = 1 if cfg.MODEL.CLS_AGNOSTIC_IOU_PRED else num_classes
        self.iou_pred = nn.Linear(representation_size, num_iou_pred_classes)

        nn.init.normal_(self.iou_pred.weight, std=0.001)
        for l in [self.iou_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        iou_pred = self.iou_pred(x)

        return iou_pred


def make_roi_iou_predictor(cfg, in_channels):
    func = registry.ROI_IOU_PREDICTOR[cfg.MODEL.ROI_IOU_HEAD.PREDICTOR]
    return func(cfg, in_channels)
