# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []

    if not cfg.SOLVER.LAPLACE_FT_MODE:
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    else:
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            if key in ['roi_heads.box_un.predictor.bbox_un_del_pred.weight',
                       'roi_heads.box_un.predictor.bbox_un_del_pred.bias',
                       'roi_heads.box_un.predictor.bbox_un_var_pred.weight',
                       'roi_heads.box_un.predictor.bbox_un_var_pred.bias']:
                lr = cfg.SOLVER.BASE_LR
            else:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.LAPLACE_LR_FACTOR

            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
