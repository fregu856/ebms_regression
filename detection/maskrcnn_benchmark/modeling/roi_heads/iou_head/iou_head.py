# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_iou_feature_extractors import make_roi_iou_feature_extractor
from .roi_iou_predictors import make_roi_iou_predictor
from .inference import make_roi_iou_post_processor
from .loss import make_roi_iou_loss_evaluator
from .loss import rect_to_rel, rel_to_rect
from maskrcnn_benchmark.structures.bounding_box import BoxList


class ROIIoUHead(torch.nn.Module):
    """
    """

    def __init__(self, cfg, in_channels):
        super(ROIIoUHead, self).__init__()
        self.feature_extractor = make_roi_iou_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_iou_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_iou_post_processor(cfg)
        self.loss_evaluator = make_roi_iou_loss_evaluator(cfg)

        self.mode = cfg.MODEL.ROI_IOU_HEAD.LOSS_TYPE
        self.cfg = cfg

    def optimize_boxes(self, features, boxes):
        # Optimize iounet boxes

        step_length = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH
        if isinstance(step_length, (tuple, list)):
            if len(step_length) == 1:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[0], step_length[0]]).to(
                    features[0].device).view(1, 4)
            elif len(step_length) == 2:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(
                    features[0].device).view(1, 4)
            else:
                raise ValueError

        if self.mode == "L2":
            box_refinement_space = "default"
        else:
            box_refinement_space = 'relative'

        box_refinement_iter = self.cfg.MODEL.ROI_IOU_HEAD.NUM_REFINE_ITER

        boxes_per_image = [b.bbox.shape[0] for b in boxes]
        step_length = [step_length.clone().expand(b.bbox.shape[0], -1).contiguous() for b in boxes]
        labels_list = [b.get_field("box_labels") for b in boxes]
        labels = torch.cat(labels_list)
        scores = [b.get_field("scores") for b in boxes]

        for f in features:
            f.requires_grad = True

        if box_refinement_space == 'default':
            # raise NotImplementedError
            # omega1 = 0.001
            # omega2 = -0.01

            for i_ in range(box_refinement_iter):
                # forward pass
                # Assume box format is xyxy
                bb_init = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes]

                for b in bb_init:
                    b.bbox.requires_grad = True

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient = torch.ones_like(iou_score))

                # Update proposal
                bb_refined = [BoxList((b.bbox + s * b.bbox.grad * (b.bbox[:, 2:] - b.bbox[:, :2]).repeat(1, 2)).detach(),
                                 b.size, b.mode) for b, s in zip(bb_init, step_length)]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                             for b_i, b_r, r_f in zip(bb_init, bb_refined, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s * (1 - r_f).float() + s * decay_factor * r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

        elif box_refinement_space == 'relative':
            boxes = [b.convert("xywh") for b in boxes]
            sz_norm = [b.bbox[:, 2:].clone() for b in boxes]

            # TODO test this
            boxes_rel = [BoxList(rect_to_rel(b.bbox, s), b.size, b.mode) for b, s in zip(boxes, sz_norm)]

            for i_ in range(box_refinement_iter):
                # forward pass
                bb_init_rel = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes_rel]

                for b in bb_init_rel:
                    b.bbox.requires_grad = True

                bb_init = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_init_rel, sz_norm)]

                bb_init = [b.convert('xyxy') for b in bb_init]

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient=torch.ones_like(iou_score))

                # Update proposal
                bb_refined_rel = [BoxList((b.bbox + s * b.bbox.grad).detach(), b.size, b.mode)
                                  for b, s in zip(bb_init_rel, step_length)]

                bb_refined = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_refined_rel, sz_norm)]
                bb_refined = [b.convert('xyxy') for b in bb_refined]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes_rel = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                                 for b_i, b_r, r_f in zip(bb_init_rel, bb_refined_rel, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s*(1 - r_f).float() + s*decay_factor*r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

            boxes = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(boxes_rel, sz_norm)]
            boxes = [b.convert("xyxy") for b in boxes]

        for b, s, l in zip(boxes, scores, labels_list):
            b.add_field("scores", s)
            b.add_field("labels", l)
            b.add_field("box_labels", l)

        return boxes

    def forward(self, features, proposals=None, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.sample_jittered_boxes(targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)

        # final classifier that converts the features into predictions
        iou_score = self.predictor(x)

        if not self.training:
            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                result = self.post_processor(proposals, iou_score)
            else:
                result = proposals
            with torch.enable_grad():
                result = self.optimize_boxes(features, result)

            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and not self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                x = self.feature_extractor(features, result)

                # final classifier that converts the features into predictions
                iou_score = self.predictor(x)

                result = self.post_processor(result, iou_score)

            return x, result, {}

        if self.training:
            loss_iou = self.loss_evaluator(
                iou_score
            )
            return (
                x,
                proposals,
                dict(loss_iou=loss_iou),
            )
        else:
            return x, iou_score, {}

class ROIIoUHead_mlis(torch.nn.Module): ############################################################################
    """
    """

    def __init__(self, cfg, in_channels):
        super(ROIIoUHead_mlis, self).__init__()
        self.feature_extractor = make_roi_iou_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_iou_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_iou_post_processor(cfg)
        self.loss_evaluator = make_roi_iou_loss_evaluator(cfg)

        self.mode = cfg.MODEL.ROI_IOU_HEAD.LOSS_TYPE
        self.cfg = cfg

    def optimize_boxes(self, features, boxes):
        # Optimize iounet boxes

        step_length = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH
        if isinstance(step_length, (tuple, list)):
            if len(step_length) == 1:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[0], step_length[0]]).to(
                    features[0].device).view(1, 4)
            elif len(step_length) == 2:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(
                    features[0].device).view(1, 4)
            else:
                raise ValueError

        if self.mode == "L2":
            box_refinement_space = "default"
        else:
            box_refinement_space = 'relative'

        box_refinement_iter = self.cfg.MODEL.ROI_IOU_HEAD.NUM_REFINE_ITER

        boxes_per_image = [b.bbox.shape[0] for b in boxes]
        step_length = [step_length.clone().expand(b.bbox.shape[0], -1).contiguous() for b in boxes]
        labels_list = [b.get_field("box_labels") for b in boxes]
        labels = torch.cat(labels_list)
        scores = [b.get_field("scores") for b in boxes]

        for f in features:
            f.requires_grad = True

        if box_refinement_space == 'default':
            # raise NotImplementedError
            # omega1 = 0.001
            # omega2 = -0.01

            for i_ in range(box_refinement_iter):
                # forward pass
                # Assume box format is xyxy
                bb_init = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes]

                for b in bb_init:
                    b.bbox.requires_grad = True

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient = torch.ones_like(iou_score))

                # Update proposal
                bb_refined = [BoxList((b.bbox + s * b.bbox.grad * (b.bbox[:, 2:] - b.bbox[:, :2]).repeat(1, 2)).detach(),
                                 b.size, b.mode) for b, s in zip(bb_init, step_length)]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                             for b_i, b_r, r_f in zip(bb_init, bb_refined, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s * (1 - r_f).float() + s * decay_factor * r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

        elif box_refinement_space == 'relative':
            boxes = [b.convert("xywh") for b in boxes]
            sz_norm = [b.bbox[:, 2:].clone() for b in boxes]

            # TODO test this
            boxes_rel = [BoxList(rect_to_rel(b.bbox, s), b.size, b.mode) for b, s in zip(boxes, sz_norm)]

            for i_ in range(box_refinement_iter):
                # forward pass
                bb_init_rel = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes_rel]

                for b in bb_init_rel:
                    b.bbox.requires_grad = True

                bb_init = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_init_rel, sz_norm)]

                bb_init = [b.convert('xyxy') for b in bb_init]

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient=torch.ones_like(iou_score))

                # Update proposal
                bb_refined_rel = [BoxList((b.bbox + s * b.bbox.grad).detach(), b.size, b.mode)
                                  for b, s in zip(bb_init_rel, step_length)]

                bb_refined = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_refined_rel, sz_norm)]
                bb_refined = [b.convert('xyxy') for b in bb_refined]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes_rel = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                                 for b_i, b_r, r_f in zip(bb_init_rel, bb_refined_rel, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s*(1 - r_f).float() + s*decay_factor*r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

            boxes = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(boxes_rel, sz_norm)]
            boxes = [b.convert("xyxy") for b in boxes]

        for b, s, l in zip(boxes, scores, labels_list):
            b.add_field("scores", s)
            b.add_field("labels", l)
            b.add_field("box_labels", l)

        return boxes

    def forward(self, features, proposals=None, targets=None, iteration=None, original_image_ids=None): ###############################################################
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # during TRAINING:
        # (features is a list of 5 elements)
        # (features[0] has shape: (16, 256, h/4, w/4)) (not always exactly h/4, w/4)
        # (features[1] has shape: (16, 256, h/8, w/8))
        # (features[2] has shape: (16, 256, h/16, w/16))
        # (features[3] has shape: (16, 256, h/32, w/32))
        # (features[4] has shape: (16, 256, h/64, w/64))
        #
        # (targets is a list of 16 elements, each element is a BoxList (e.g. [BoxList(num_boxes=3, image_width=800, image_height=1066, mode=xyxy), BoxList(num_boxes=19, image_width=800, image_height=1201, mode=xyxy),...]))

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.sample_jittered_boxes(targets) #######################################################
                # (proposals is a list of 16 elements, each element is a BoxList, num_boxes in each BoxList is M*{num_boxes for the corresponding BoxList in targets})

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # (x has shape: (num_preds, 1024)) (num_preds is different from batch to batch, e.g. 12032 or 19072 or 20992)

        # final classifier that converts the features into predictions
        iou_score = self.predictor(x)
        # (iou_score has shape: (num_preds, 81)) (81 is the number of classes)

        if not self.training:
            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                result = self.post_processor(proposals, iou_score)
            else:
                result = proposals
            with torch.enable_grad():
                result = self.optimize_boxes(features, result)

            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and not self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                x = self.feature_extractor(features, result)

                # final classifier that converts the features into predictions
                iou_score = self.predictor(x)

                result = self.post_processor(result, iou_score)

            return x, result, {}

        if self.training:
            loss_iou = self.loss_evaluator(
                iou_score
            )
            # (loss_iou is just a tensor of a single value)
            return (
                x,
                proposals,
                dict(loss_iou=loss_iou),
            )
        else:
            return x, iou_score, {}

class ROIIoUHead_kldis(torch.nn.Module): ############################################################################
    """
    """

    def __init__(self, cfg, in_channels):
        super(ROIIoUHead_kldis, self).__init__()
        self.feature_extractor = make_roi_iou_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_iou_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_iou_post_processor(cfg)
        self.loss_evaluator = make_roi_iou_loss_evaluator(cfg)

        self.mode = cfg.MODEL.ROI_IOU_HEAD.LOSS_TYPE
        self.cfg = cfg

    def optimize_boxes(self, features, boxes):
        # Optimize iounet boxes

        step_length = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH
        if isinstance(step_length, (tuple, list)):
            if len(step_length) == 1:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[0], step_length[0]]).to(
                    features[0].device).view(1, 4)
            elif len(step_length) == 2:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(
                    features[0].device).view(1, 4)
            else:
                raise ValueError

        if self.mode == "L2":
            box_refinement_space = "default"
        else:
            box_refinement_space = 'relative'

        box_refinement_iter = self.cfg.MODEL.ROI_IOU_HEAD.NUM_REFINE_ITER

        boxes_per_image = [b.bbox.shape[0] for b in boxes]
        step_length = [step_length.clone().expand(b.bbox.shape[0], -1).contiguous() for b in boxes]
        labels_list = [b.get_field("box_labels") for b in boxes]
        labels = torch.cat(labels_list)
        scores = [b.get_field("scores") for b in boxes]

        for f in features:
            f.requires_grad = True

        if box_refinement_space == 'default':
            # raise NotImplementedError
            # omega1 = 0.001
            # omega2 = -0.01

            for i_ in range(box_refinement_iter):
                # forward pass
                # Assume box format is xyxy
                bb_init = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes]

                for b in bb_init:
                    b.bbox.requires_grad = True

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient = torch.ones_like(iou_score))

                # Update proposal
                bb_refined = [BoxList((b.bbox + s * b.bbox.grad * (b.bbox[:, 2:] - b.bbox[:, :2]).repeat(1, 2)).detach(),
                                 b.size, b.mode) for b, s in zip(bb_init, step_length)]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                             for b_i, b_r, r_f in zip(bb_init, bb_refined, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s * (1 - r_f).float() + s * decay_factor * r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

        elif box_refinement_space == 'relative':
            boxes = [b.convert("xywh") for b in boxes]
            sz_norm = [b.bbox[:, 2:].clone() for b in boxes]

            # TODO test this
            boxes_rel = [BoxList(rect_to_rel(b.bbox, s), b.size, b.mode) for b, s in zip(boxes, sz_norm)]

            for i_ in range(box_refinement_iter):
                # forward pass
                bb_init_rel = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes_rel]

                for b in bb_init_rel:
                    b.bbox.requires_grad = True

                bb_init = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_init_rel, sz_norm)]

                bb_init = [b.convert('xyxy') for b in bb_init]

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient=torch.ones_like(iou_score))

                # Update proposal
                bb_refined_rel = [BoxList((b.bbox + s * b.bbox.grad).detach(), b.size, b.mode)
                                  for b, s in zip(bb_init_rel, step_length)]

                bb_refined = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_refined_rel, sz_norm)]
                bb_refined = [b.convert('xyxy') for b in bb_refined]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes_rel = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                                 for b_i, b_r, r_f in zip(bb_init_rel, bb_refined_rel, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s*(1 - r_f).float() + s*decay_factor*r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

            boxes = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(boxes_rel, sz_norm)]
            boxes = [b.convert("xyxy") for b in boxes]

        for b, s, l in zip(boxes, scores, labels_list):
            b.add_field("scores", s)
            b.add_field("labels", l)
            b.add_field("box_labels", l)

        return boxes

    def forward(self, features, proposals=None, targets=None, iteration=None, original_image_ids=None): ###############################################################
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # during TRAINING:
        # (features is a list of 5 elements)
        # (features[0] has shape: (16, 256, h/4, w/4)) (not always exactly h/4, w/4)
        # (features[1] has shape: (16, 256, h/8, w/8))
        # (features[2] has shape: (16, 256, h/16, w/16))
        # (features[3] has shape: (16, 256, h/32, w/32))
        # (features[4] has shape: (16, 256, h/64, w/64))
        #
        # (targets is a list of 16 elements, each element is a BoxList (e.g. [BoxList(num_boxes=3, image_width=800, image_height=1066, mode=xyxy), BoxList(num_boxes=19, image_width=800, image_height=1201, mode=xyxy),...]))

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.sample_jittered_boxes(targets) #######################################################
                # (proposals is a list of 16 elements, each element is a BoxList, num_boxes in each BoxList is M*{num_boxes for the corresponding BoxList in targets})

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # (x has shape: (num_preds, 1024)) (num_preds is different from batch to batch, e.g. 12032 or 19072 or 20992)

        # final classifier that converts the features into predictions
        iou_score = self.predictor(x)
        # (iou_score has shape: (num_preds, 81)) (81 is the number of classes)

        if not self.training:
            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                result = self.post_processor(proposals, iou_score)
            else:
                result = proposals
            with torch.enable_grad():
                result = self.optimize_boxes(features, result)

            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and not self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                x = self.feature_extractor(features, result)

                # final classifier that converts the features into predictions
                iou_score = self.predictor(x)

                result = self.post_processor(result, iou_score)

            return x, result, {}

        if self.training:
            loss_iou = self.loss_evaluator(
                iou_score
            )
            # (loss_iou is just a tensor of a single value)
            return (
                x,
                proposals,
                dict(loss_iou=loss_iou),
            )
        else:
            return x, iou_score, {}

class ROIIoUHead_nce(torch.nn.Module): ############################################################################
    """
    """

    def __init__(self, cfg, in_channels):
        super(ROIIoUHead_nce, self).__init__()
        self.feature_extractor = make_roi_iou_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_iou_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_iou_post_processor(cfg)
        self.loss_evaluator = make_roi_iou_loss_evaluator(cfg)

        self.mode = cfg.MODEL.ROI_IOU_HEAD.LOSS_TYPE
        self.cfg = cfg

    def optimize_boxes(self, features, boxes):
        # Optimize iounet boxes

        step_length = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH
        if isinstance(step_length, (tuple, list)):
            if len(step_length) == 1:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[0], step_length[0]]).to(
                    features[0].device).view(1, 4)
            elif len(step_length) == 2:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(
                    features[0].device).view(1, 4)
            else:
                raise ValueError

        if self.mode == "L2":
            box_refinement_space = "default"
        else:
            box_refinement_space = 'relative'

        box_refinement_iter = self.cfg.MODEL.ROI_IOU_HEAD.NUM_REFINE_ITER

        boxes_per_image = [b.bbox.shape[0] for b in boxes]
        step_length = [step_length.clone().expand(b.bbox.shape[0], -1).contiguous() for b in boxes]
        labels_list = [b.get_field("box_labels") for b in boxes]
        labels = torch.cat(labels_list)
        scores = [b.get_field("scores") for b in boxes]

        for f in features:
            f.requires_grad = True

        if box_refinement_space == 'default':
            # raise NotImplementedError
            # omega1 = 0.001
            # omega2 = -0.01

            for i_ in range(box_refinement_iter):
                # forward pass
                # Assume box format is xyxy
                bb_init = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes]

                for b in bb_init:
                    b.bbox.requires_grad = True

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient = torch.ones_like(iou_score))

                # Update proposal
                bb_refined = [BoxList((b.bbox + s * b.bbox.grad * (b.bbox[:, 2:] - b.bbox[:, :2]).repeat(1, 2)).detach(),
                                 b.size, b.mode) for b, s in zip(bb_init, step_length)]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                             for b_i, b_r, r_f in zip(bb_init, bb_refined, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s * (1 - r_f).float() + s * decay_factor * r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

        elif box_refinement_space == 'relative':
            boxes = [b.convert("xywh") for b in boxes]
            sz_norm = [b.bbox[:, 2:].clone() for b in boxes]

            # TODO test this
            boxes_rel = [BoxList(rect_to_rel(b.bbox, s), b.size, b.mode) for b, s in zip(boxes, sz_norm)]

            for i_ in range(box_refinement_iter):
                # forward pass
                bb_init_rel = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes_rel]

                for b in bb_init_rel:
                    b.bbox.requires_grad = True

                bb_init = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_init_rel, sz_norm)]

                bb_init = [b.convert('xyxy') for b in bb_init]

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient=torch.ones_like(iou_score))

                # Update proposal
                bb_refined_rel = [BoxList((b.bbox + s * b.bbox.grad).detach(), b.size, b.mode)
                                  for b, s in zip(bb_init_rel, step_length)]

                bb_refined = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_refined_rel, sz_norm)]
                bb_refined = [b.convert('xyxy') for b in bb_refined]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes_rel = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                                 for b_i, b_r, r_f in zip(bb_init_rel, bb_refined_rel, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s*(1 - r_f).float() + s*decay_factor*r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

            boxes = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(boxes_rel, sz_norm)]
            boxes = [b.convert("xyxy") for b in boxes]

        for b, s, l in zip(boxes, scores, labels_list):
            b.add_field("scores", s)
            b.add_field("labels", l)
            b.add_field("box_labels", l)

        return boxes

    def forward(self, features, proposals=None, targets=None, iteration=None, original_image_ids=None): ###############################################################
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # during TRAINING:
        # (features is a list of 5 elements)
        # (features[0] has shape: (16, 256, h/4, w/4)) (not always exactly h/4, w/4)
        # (features[1] has shape: (16, 256, h/8, w/8))
        # (features[2] has shape: (16, 256, h/16, w/16))
        # (features[3] has shape: (16, 256, h/32, w/32))
        # (features[4] has shape: (16, 256, h/64, w/64))
        #
        # (targets is a list of 16 elements, each element is a BoxList (e.g. [BoxList(num_boxes=3, image_width=800, image_height=1066, mode=xyxy), BoxList(num_boxes=19, image_width=800, image_height=1201, mode=xyxy),...]))

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.sample_jittered_boxes(targets) #######################################################
                # (proposals is a list of 16 elements, each element is a BoxList, num_boxes in each BoxList is M*{num_boxes for the corresponding BoxList in targets})

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # (x has shape: (num_preds, 1024)) (num_preds is different from batch to batch, e.g. 12032 or 19072 or 20992)

        # final classifier that converts the features into predictions
        iou_score = self.predictor(x)
        # (iou_score has shape: (num_preds, 81)) (81 is the number of classes)

        if not self.training:
            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                result = self.post_processor(proposals, iou_score)
            else:
                result = proposals
            with torch.enable_grad():
                result = self.optimize_boxes(features, result)

            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and not self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                x = self.feature_extractor(features, result)

                # final classifier that converts the features into predictions
                iou_score = self.predictor(x)

                result = self.post_processor(result, iou_score)

            return x, result, {}

        if self.training:
            loss_iou = self.loss_evaluator(
                iou_score
            )
            # (loss_iou is just a tensor of a single value)
            return (
                x,
                proposals,
                dict(loss_iou=loss_iou),
            )
        else:
            return x, iou_score, {}

class ROIIoUHead_dsm(torch.nn.Module): ############################################################################
    """
    """

    def __init__(self, cfg, in_channels):
        super(ROIIoUHead_dsm, self).__init__()
        self.feature_extractor = make_roi_iou_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_iou_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_iou_post_processor(cfg)
        self.loss_evaluator = make_roi_iou_loss_evaluator(cfg)

        self.mode = cfg.MODEL.ROI_IOU_HEAD.LOSS_TYPE
        self.cfg = cfg

    def optimize_boxes(self, features, boxes):
        # Optimize iounet boxes

        step_length = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH
        if isinstance(step_length, (tuple, list)):
            if len(step_length) == 1:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[0], step_length[0]]).to(
                    features[0].device).view(1, 4)
            elif len(step_length) == 2:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(
                    features[0].device).view(1, 4)
            else:
                raise ValueError

        if self.mode == "L2":
            box_refinement_space = "default"
        else:
            box_refinement_space = 'relative'

        box_refinement_iter = self.cfg.MODEL.ROI_IOU_HEAD.NUM_REFINE_ITER

        boxes_per_image = [b.bbox.shape[0] for b in boxes]
        step_length = [step_length.clone().expand(b.bbox.shape[0], -1).contiguous() for b in boxes]
        labels_list = [b.get_field("box_labels") for b in boxes]
        labels = torch.cat(labels_list)
        scores = [b.get_field("scores") for b in boxes]

        for f in features:
            f.requires_grad = True

        if box_refinement_space == 'default':
            # raise NotImplementedError
            # omega1 = 0.001
            # omega2 = -0.01

            for i_ in range(box_refinement_iter):
                # forward pass
                # Assume box format is xyxy
                bb_init = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes]

                for b in bb_init:
                    b.bbox.requires_grad = True

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient = torch.ones_like(iou_score))

                # Update proposal
                bb_refined = [BoxList((b.bbox + s * b.bbox.grad * (b.bbox[:, 2:] - b.bbox[:, :2]).repeat(1, 2)).detach(),
                                 b.size, b.mode) for b, s in zip(bb_init, step_length)]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                             for b_i, b_r, r_f in zip(bb_init, bb_refined, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s * (1 - r_f).float() + s * decay_factor * r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

        elif box_refinement_space == 'relative':
            boxes = [b.convert("xywh") for b in boxes]
            sz_norm = [b.bbox[:, 2:].clone() for b in boxes]

            # TODO test this
            boxes_rel = [BoxList(rect_to_rel(b.bbox, s), b.size, b.mode) for b, s in zip(boxes, sz_norm)]

            for i_ in range(box_refinement_iter):
                # forward pass
                bb_init_rel = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes_rel]

                for b in bb_init_rel:
                    b.bbox.requires_grad = True

                bb_init = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_init_rel, sz_norm)]

                bb_init = [b.convert('xyxy') for b in bb_init]

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient=torch.ones_like(iou_score))

                # Update proposal
                bb_refined_rel = [BoxList((b.bbox + s * b.bbox.grad).detach(), b.size, b.mode)
                                  for b, s in zip(bb_init_rel, step_length)]

                bb_refined = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_refined_rel, sz_norm)]
                bb_refined = [b.convert('xyxy') for b in bb_refined]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes_rel = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                                 for b_i, b_r, r_f in zip(bb_init_rel, bb_refined_rel, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s*(1 - r_f).float() + s*decay_factor*r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

            boxes = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(boxes_rel, sz_norm)]
            boxes = [b.convert("xyxy") for b in boxes]

        for b, s, l in zip(boxes, scores, labels_list):
            b.add_field("scores", s)
            b.add_field("labels", l)
            b.add_field("box_labels", l)

        return boxes

    def forward(self, features, proposals=None, targets=None, iteration=None, original_image_ids=None): ###############################################################
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # during TRAINING:
        # (features is a list of 5 elements)
        # (features[0] has shape: (16, 256, h/4, w/4)) (not always exactly h/4, w/4)
        # (features[1] has shape: (16, 256, h/8, w/8))
        # (features[2] has shape: (16, 256, h/16, w/16))
        # (features[3] has shape: (16, 256, h/32, w/32))
        # (features[4] has shape: (16, 256, h/64, w/64))
        #
        # (targets is a list of 16 elements, each element is a BoxList (e.g. [BoxList(num_boxes=3, image_width=800, image_height=1066, mode=xyxy), BoxList(num_boxes=19, image_width=800, image_height=1201, mode=xyxy),...]))

        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.sample_jittered_boxes(targets) #######################################################
                # (proposals is a list of 16 elements, each element is a BoxList, num_boxes in each BoxList is M*{num_boxes for the corresponding BoxList in targets})

            for proposal in proposals:
                proposal.bbox.requires_grad = True

        x = self.feature_extractor(features, proposals)
        iou_score = self.predictor(x)
        # (shape: (num_gt_bboxes_in_batch*M, 81)) (81 is the number of classes)

        if not self.training:
            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                result = self.post_processor(proposals, iou_score)
            else:
                result = proposals
            with torch.enable_grad():
                result = self.optimize_boxes(features, result)

            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and not self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                x = self.feature_extractor(features, result)

                # final classifier that converts the features into predictions
                iou_score = self.predictor(x)

                result = self.post_processor(result, iou_score)

            return x, result, {}

        if self.training:
            loss_iou = self.loss_evaluator(iou_score, targets, proposals)
            # (loss_iou is just a tensor of a single value)
            return (
                x,
                proposals,
                dict(loss_iou=loss_iou),
            )
        else:
            return x, iou_score, {}

class ROIIoUHead_mlmcmc(torch.nn.Module): ############################################################################
    """
    """

    def __init__(self, cfg, in_channels):
        super(ROIIoUHead_mlmcmc, self).__init__()
        self.feature_extractor = make_roi_iou_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_iou_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_iou_post_processor(cfg)
        self.loss_evaluator = make_roi_iou_loss_evaluator(cfg)

        self.mode = cfg.MODEL.ROI_IOU_HEAD.LOSS_TYPE
        self.cfg = cfg

        self.L = cfg.MODEL.ROI_IOU_HEAD.LANGEVIN_L
        self.alpha = cfg.MODEL.ROI_IOU_HEAD.LANGEVIN_alpha

    def optimize_boxes(self, features, boxes):
        # Optimize iounet boxes

        step_length = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH
        if isinstance(step_length, (tuple, list)):
            if len(step_length) == 1:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[0], step_length[0]]).to(
                    features[0].device).view(1, 4)
            elif len(step_length) == 2:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(
                    features[0].device).view(1, 4)
            else:
                raise ValueError

        if self.mode == "L2":
            box_refinement_space = "default"
        else:
            box_refinement_space = 'relative'

        box_refinement_iter = self.cfg.MODEL.ROI_IOU_HEAD.NUM_REFINE_ITER

        boxes_per_image = [b.bbox.shape[0] for b in boxes]
        step_length = [step_length.clone().expand(b.bbox.shape[0], -1).contiguous() for b in boxes]
        labels_list = [b.get_field("box_labels") for b in boxes]
        labels = torch.cat(labels_list)
        scores = [b.get_field("scores") for b in boxes]

        for f in features:
            f.requires_grad = True

        if box_refinement_space == 'default':
            # raise NotImplementedError
            # omega1 = 0.001
            # omega2 = -0.01

            for i_ in range(box_refinement_iter):
                # forward pass
                # Assume box format is xyxy
                bb_init = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes]

                for b in bb_init:
                    b.bbox.requires_grad = True

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient = torch.ones_like(iou_score))

                # Update proposal
                bb_refined = [BoxList((b.bbox + s * b.bbox.grad * (b.bbox[:, 2:] - b.bbox[:, :2]).repeat(1, 2)).detach(),
                                 b.size, b.mode) for b, s in zip(bb_init, step_length)]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                             for b_i, b_r, r_f in zip(bb_init, bb_refined, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s * (1 - r_f).float() + s * decay_factor * r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

        elif box_refinement_space == 'relative':
            boxes = [b.convert("xywh") for b in boxes]
            sz_norm = [b.bbox[:, 2:].clone() for b in boxes]

            # TODO test this
            boxes_rel = [BoxList(rect_to_rel(b.bbox, s), b.size, b.mode) for b, s in zip(boxes, sz_norm)]

            for i_ in range(box_refinement_iter):
                # forward pass
                bb_init_rel = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes_rel]

                for b in bb_init_rel:
                    b.bbox.requires_grad = True

                bb_init = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_init_rel, sz_norm)]

                bb_init = [b.convert('xyxy') for b in bb_init]

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient=torch.ones_like(iou_score))

                # Update proposal
                bb_refined_rel = [BoxList((b.bbox + s * b.bbox.grad).detach(), b.size, b.mode)
                                  for b, s in zip(bb_init_rel, step_length)]

                bb_refined = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_refined_rel, sz_norm)]
                bb_refined = [b.convert('xyxy') for b in bb_refined]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes_rel = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                                 for b_i, b_r, r_f in zip(bb_init_rel, bb_refined_rel, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s*(1 - r_f).float() + s*decay_factor*r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

            boxes = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(boxes_rel, sz_norm)]
            boxes = [b.convert("xyxy") for b in boxes]

        for b, s, l in zip(boxes, scores, labels_list):
            b.add_field("scores", s)
            b.add_field("labels", l)
            b.add_field("box_labels", l)

        return boxes

    def forward(self, features, proposals=None, targets=None, iteration=None, original_image_ids=None): ###############################################################
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # during TRAINING:
        # (features is a list of 5 elements)
        # (features[0] has shape: (16, 256, h/4, w/4)) (not always exactly h/4, w/4)
        # (features[1] has shape: (16, 256, h/8, w/8))
        # (features[2] has shape: (16, 256, h/16, w/16))
        # (features[3] has shape: (16, 256, h/32, w/32))
        # (features[4] has shape: (16, 256, h/64, w/64))
        #
        # (targets is a list of 16 elements, each element is a BoxList (e.g. [BoxList(num_boxes=3, image_width=800, image_height=1066, mode=xyxy), BoxList(num_boxes=19, image_width=800, image_height=1201, mode=xyxy),...]))

        print (self.L)
        print (self.alpha)

        if self.training:
            target_labels_list = [b.get_field("labels") for b in targets]
            target_labels = torch.cat(target_labels_list).long() # (shape: (num_gt_bboxes_in_batch))
            print (target_labels.size())

            fs = self.predictor(self.feature_extractor(features, targets)) # (shape: (num_gt_bboxes_in_batch, 81)) (81 is the number of classes)
            # print (fs.size())
            fs = fs[torch.arange(fs.shape[0]), target_labels] # (shape: (num_gt_bboxes_in_batch))
            # print (fs.size())

        if self.training:
            proposals = self.loss_evaluator.copy_targets(targets)
            # (proposals is a list of 16 elements, each element is a BoxList, num_boxes in each BoxList is M*{num_boxes for the corresponding BoxList in targets})
            # (proposals are just M copies of each target)

            proposal_labels_list = [b.get_field("labels") for b in proposals]
            proposal_labels = torch.cat(proposal_labels_list).long() # (shape: (num_gt_bboxes_in_batch*M))
            # print (proposal_labels.size())

            proposals = [b.convert("xywh") for b in proposals]
            sz_norm = [b.bbox[:, 2:].clone() for b in proposals]

            # print (proposals[0].bbox[0:10])
            # print ("@@@@@@@@@@@@@@@@@")

            proposals_rel = [BoxList(rect_to_rel(b.bbox, s), b.size, b.mode) for b, s in zip(proposals, sz_norm)]
            # print (proposals_rel[0].bbox[0:10])
            for l in range(self.L):
                # print (l)

                bb_init_rel = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in proposals_rel]
                for b in bb_init_rel:
                    b.bbox.requires_grad = True
                bb_init = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_init_rel, sz_norm)]
                bb_init = [b.convert('xyxy') for b in bb_init]

                f_proposals = self.predictor(self.feature_extractor(features, bb_init)) # (shape: (num_gt_bboxes_in_batch*M, 81))
                # print (f_proposals.size())
                f_proposals = f_proposals[torch.arange(f_proposals.shape[0]), proposal_labels] # (shape: (num_gt_bboxes_in_batch*M))
                # print (f_proposals.size())

                f_proposals.backward(gradient=torch.ones_like(f_proposals))

                # print (bb_init_rel[0].bbox.grad[0:10])
                proposals_rel = [BoxList((b.bbox + (0.5*self.alpha**2)*b.bbox.grad).detach() + self.alpha*torch.randn(b.bbox.size()).cuda(), b.size, b.mode) for b in bb_init_rel]
                # print (proposals_rel[0].bbox[0:10])

            proposals = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(proposals_rel, sz_norm)]
            # print ("@@@@@@@@@@@@@@@@@")
            # print (proposals[0].bbox[0:10])
            proposals = [b.convert("xyxy") for b in proposals]

            for b, l in zip(proposals, proposal_labels_list):
                b.add_field("labels", l)

        if self.training:
            proposal_labels_list = [b.get_field("labels") for b in proposals]
            proposal_labels = torch.cat(proposal_labels_list).long() # (shape: (num_gt_bboxes_in_batch*M))
            # print (proposal_labels.size())

            x = self.feature_extractor(features, proposals)
            f_samples = self.predictor(x)
            # (shape: (num_gt_bboxes_in_batch*M, 81)) (81 is the number of classes)
            # print (f_samples.size())
            f_samples = f_samples[torch.arange(f_samples.shape[0]), proposal_labels] # (shape: (num_gt_bboxes_in_batch*M))
            # print (f_samples.size())
        else:
            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            x = self.feature_extractor(features, proposals)
            # (x has shape: (num_preds, 1024)) (num_preds is different from batch to batch, e.g. 12032 or 19072 or 20992)
            # (num_preds == num_gt_bboxes_in_batch*M) (M == 128)

            # final classifier that converts the features into predictions
            iou_score = self.predictor(x)
            # (iou_score has shape: (num_preds, 81)) (81 is the number of classes)

        if not self.training:
            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                result = self.post_processor(proposals, iou_score)
            else:
                result = proposals
            with torch.enable_grad():
                result = self.optimize_boxes(features, result)

            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and not self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                x = self.feature_extractor(features, result)

                # final classifier that converts the features into predictions
                iou_score = self.predictor(x)

                result = self.post_processor(result, iou_score)

            return x, result, {}

        if self.training:
            loss_iou = self.loss_evaluator(fs, f_samples, target_labels)
            # (loss_iou is just a tensor of a single value)
            return (
                x,
                proposals,
                dict(loss_iou=loss_iou),
            )

        else:
            return x, iou_score, {}

class ROIIoUHead_nceplus(torch.nn.Module): ############################################################################
    """
    """

    def __init__(self, cfg, in_channels):
        super(ROIIoUHead_nceplus, self).__init__()
        self.feature_extractor = make_roi_iou_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_iou_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_iou_post_processor(cfg)
        self.loss_evaluator = make_roi_iou_loss_evaluator(cfg)

        self.mode = cfg.MODEL.ROI_IOU_HEAD.LOSS_TYPE
        self.cfg = cfg

        self.beta = cfg.MODEL.ROI_IOU_HEAD.NCEPLUS_beta
        print (self.beta)

    def optimize_boxes(self, features, boxes):
        # Optimize iounet boxes

        step_length = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH
        if isinstance(step_length, (tuple, list)):
            if len(step_length) == 1:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[0], step_length[0]]).to(
                    features[0].device).view(1, 4)
            elif len(step_length) == 2:
                step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(
                    features[0].device).view(1, 4)
            else:
                raise ValueError

        if self.mode == "L2":
            box_refinement_space = "default"
        else:
            box_refinement_space = 'relative'

        box_refinement_iter = self.cfg.MODEL.ROI_IOU_HEAD.NUM_REFINE_ITER

        boxes_per_image = [b.bbox.shape[0] for b in boxes]
        step_length = [step_length.clone().expand(b.bbox.shape[0], -1).contiguous() for b in boxes]
        labels_list = [b.get_field("box_labels") for b in boxes]
        labels = torch.cat(labels_list)
        scores = [b.get_field("scores") for b in boxes]

        for f in features:
            f.requires_grad = True

        if box_refinement_space == 'default':
            # raise NotImplementedError
            # omega1 = 0.001
            # omega2 = -0.01

            for i_ in range(box_refinement_iter):
                # forward pass
                # Assume box format is xyxy
                bb_init = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes]

                for b in bb_init:
                    b.bbox.requires_grad = True

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient = torch.ones_like(iou_score))

                # Update proposal
                bb_refined = [BoxList((b.bbox + s * b.bbox.grad * (b.bbox[:, 2:] - b.bbox[:, :2]).repeat(1, 2)).detach(),
                                 b.size, b.mode) for b, s in zip(bb_init, step_length)]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                             for b_i, b_r, r_f in zip(bb_init, bb_refined, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s * (1 - r_f).float() + s * decay_factor * r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

        elif box_refinement_space == 'relative':
            boxes = [b.convert("xywh") for b in boxes]
            sz_norm = [b.bbox[:, 2:].clone() for b in boxes]

            # TODO test this
            boxes_rel = [BoxList(rect_to_rel(b.bbox, s), b.size, b.mode) for b, s in zip(boxes, sz_norm)]

            for i_ in range(box_refinement_iter):
                # forward pass
                bb_init_rel = [BoxList(b.bbox.clone().detach(), b.size, b.mode) for b in boxes_rel]

                for b in bb_init_rel:
                    b.bbox.requires_grad = True

                bb_init = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_init_rel, sz_norm)]

                bb_init = [b.convert('xyxy') for b in bb_init]

                x = self.feature_extractor(features, bb_init)

                iou_score = self.predictor(x)
                iou_score = iou_score[torch.arange(iou_score.shape[0]), labels]

                iou_score.backward(gradient=torch.ones_like(iou_score))

                # Update proposal
                bb_refined_rel = [BoxList((b.bbox + s * b.bbox.grad).detach(), b.size, b.mode)
                                  for b, s in zip(bb_init_rel, step_length)]

                bb_refined = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(bb_refined_rel, sz_norm)]
                bb_refined = [b.convert('xyxy') for b in bb_refined]

                with torch.no_grad():
                    x = self.feature_extractor(features, bb_refined)

                    new_iou_score = self.predictor(x)
                    new_iou_score = new_iou_score[torch.arange(new_iou_score.shape[0]), labels]

                    refinement_failed = (new_iou_score < iou_score)
                    refinement_failed = refinement_failed.view(-1, 1)
                    refinement_failed = refinement_failed.split(boxes_per_image, dim=0)

                    boxes_rel = [BoxList(b_i.bbox * r_f.float() + b_r.bbox * (1 - r_f).float(), b_i.size, b_i.mode)
                                 for b_i, b_r, r_f in zip(bb_init_rel, bb_refined_rel, refinement_failed)]

                    # decay step length for failures
                    decay_factor = self.cfg.MODEL.ROI_IOU_HEAD.STEP_LENGTH_DECAY
                    step_length = [s*(1 - r_f).float() + s*decay_factor*r_f.float()
                                   for s, r_f in zip(step_length, refinement_failed)]

            boxes = [BoxList(rel_to_rect(b.bbox, s), b.size, b.mode) for b, s in zip(boxes_rel, sz_norm)]
            boxes = [b.convert("xyxy") for b in boxes]

        for b, s, l in zip(boxes, scores, labels_list):
            b.add_field("scores", s)
            b.add_field("labels", l)
            b.add_field("box_labels", l)

        return boxes

    def forward(self, features, proposals=None, targets=None, iteration=None, original_image_ids=None): ###############################################################
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        # during TRAINING:
        # (features is a list of 5 elements)
        # (features[0] has shape: (16, 256, h/4, w/4)) (not always exactly h/4, w/4)
        # (features[1] has shape: (16, 256, h/8, w/8))
        # (features[2] has shape: (16, 256, h/16, w/16))
        # (features[3] has shape: (16, 256, h/32, w/32))
        # (features[4] has shape: (16, 256, h/64, w/64))
        #
        # (targets is a list of 16 elements, each element is a BoxList (e.g. [BoxList(num_boxes=3, image_width=800, image_height=1066, mode=xyxy), BoxList(num_boxes=19, image_width=800, image_height=1201, mode=xyxy),...]))

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.sample_jittered_boxes(targets, self.beta) #######################################################
                # (proposals is a list of 16 elements, each element is a BoxList, num_boxes in each BoxList is M*{num_boxes for the corresponding BoxList in targets})

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # (x has shape: (num_preds, 1024)) (num_preds is different from batch to batch, e.g. 12032 or 19072 or 20992)

        # final classifier that converts the features into predictions
        iou_score = self.predictor(x)
        # (iou_score has shape: (num_preds, 81)) (81 is the number of classes)

        if not self.training:
            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                result = self.post_processor(proposals, iou_score)
            else:
                result = proposals
            with torch.enable_grad():
                result = self.optimize_boxes(features, result)

            if self.cfg.MODEL.ROI_IOU_HEAD.PERFORM_FILTERING and not self.cfg.MODEL.ROI_IOU_HEAD.NMS_BEFORE:
                x = self.feature_extractor(features, result)

                # final classifier that converts the features into predictions
                iou_score = self.predictor(x)

                result = self.post_processor(result, iou_score)

            return x, result, {}

        if self.training:
            loss_iou = self.loss_evaluator(iou_score)
            # (loss_iou is just a tensor of a single value)
            return (
                x,
                proposals,
                dict(loss_iou=loss_iou),
            )
        else:
            return x, iou_score, {}


def build_roi_iou_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """

    loss_type = cfg.MODEL.ROI_IOU_HEAD.LOSS_TYPE

    if loss_type == "ML-IS":
        return ROIIoUHead_mlis(cfg, in_channels)

    elif loss_type == "KLD-IS":
        return ROIIoUHead_kldis(cfg, in_channels)

    elif loss_type == "NCE":
        return ROIIoUHead_nce(cfg, in_channels)

    elif loss_type == "DSM":
        return ROIIoUHead_dsm(cfg, in_channels)

    elif loss_type == "ML-MCMC":
        return ROIIoUHead_mlmcmc(cfg, in_channels)

    elif loss_type == "NCE+":
        return ROIIoUHead_nceplus(cfg, in_channels)

    else:
        raise ValueError
