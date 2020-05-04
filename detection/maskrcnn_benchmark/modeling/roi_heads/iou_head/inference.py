# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        cls_agnostic_iou_pred=False,
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img

        assert cls_agnostic_iou_pred is False
        self.cls_agnostic_iou_pred = cls_agnostic_iou_pred

    def forward(self, boxes, iou_scores):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """

        boxes_per_image = [len(box) for box in boxes]
        iou_scores = iou_scores.split(boxes_per_image, dim=0)

        results = []
        for b, b_iou in zip(boxes, iou_scores):
            boxlist = self.filter_results(b, b_iou)
            results.append(boxlist)
        return results

    def filter_results(self, boxlist, iou_scores):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        # TODO
        num_classes = 81

        boxes = boxlist.bbox
        scores = boxlist.get_field("scores")
        box_labels = boxlist.get_field("box_labels")

        device = scores.device
        result = []

        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        for j in range(1, num_classes):
            inds = (box_labels == j).nonzero().squeeze()
            scores_j = scores[inds]
            iou_scores_j = iou_scores[inds, j]

            boxes_j = boxes[inds, :].view(-1, 4)
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class.add_field("iou_scores", iou_scores_j)
            boxlist_for_class = boxlist_nms_iou(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "box_labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]

        return result


def make_roi_iou_post_processor(cfg):
    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_iou_pred = cfg.MODEL.CLS_AGNOSTIC_IOU_PRED

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        cls_agnostic_iou_pred,
    )
    return postprocessor
