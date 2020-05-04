# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn


@registry.ROI_BOX_UN_PREDICTOR.register("FPNUnPredictor")
class FPNUnPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNUnPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_UN_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score_pred = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes

        self.bbox_un_del_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.bbox_un_var_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score_pred.weight, std=0.01)
        nn.init.normal_(self.bbox_un_del_pred.weight, std=0.001)
        nn.init.normal_(self.bbox_un_var_pred.weight, std=0.001)
        for l in [self.cls_score_pred, self.bbox_un_del_pred, self.bbox_un_var_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score_pred(x)
        bbox_deltas = self.bbox_un_del_pred(x)
        bbox_un_pred = self.bbox_un_var_pred(x)

        return scores, bbox_deltas, bbox_un_pred


@registry.ROI_BOX_UN_PREDICTOR.register("FPNUnPredictorSh")
class FPNUnPredictorSh(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNUnPredictorSh, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_UN_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes

        self.bbox_un_del_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.bbox_un_var_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_un_del_pred.weight, std=0.001)
        nn.init.normal_(self.bbox_un_var_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_un_del_pred, self.bbox_un_var_pred]:
            nn.init.constant_(l.bias, 0)

        if not cfg.MODEL.ROI_BOX_UN_HEAD.TRAIN_HEAD:
            for p in self.cls_score.parameters():
                p.requires_grad = False

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_un_del_pred(x)
        bbox_un_pred = self.bbox_un_var_pred(x)

        return scores, bbox_deltas, bbox_un_pred

def make_roi_box_un_predictor(cfg, in_channels):
    func = registry.ROI_BOX_UN_PREDICTOR[cfg.MODEL.ROI_BOX_UN_HEAD.PREDICTOR]
    net = func(cfg, in_channels)

    return net
