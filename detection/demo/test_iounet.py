# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
import torch

import time


def main():
    # config_file = "../configs/iounet_ft_test4.yaml"
    config_file = "../configs/iounet_ft_ml_test2.yaml"
    # config_file = "../configs/test.yaml"
    # image_path = "/home/goutam/data/tracking_datasets/coco/val2017/000000000785.jpg"
    # init_box = [[190, 20, 500, 330], [240, 50, 450, 280]]
    image_path = "/home/goutam/data/tracking_datasets/coco/val2017/000000015440.jpg"
    box = [80, 80, 260, 260]
    init_box = [box, box]
    # load config from file and command-line arguments
    cfg.merge_from_file(config_file)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
    )

    im = cv2.imread(image_path)

    im_disp = im.copy()

    image = coco_demo.transforms(im)

    image_list = to_image_list(image, coco_demo.cfg.DATALOADER.SIZE_DIVISIBILITY)
    image_list = image_list.to(coco_demo.device)

    label = 12
    proposal = BoxList(torch.tensor(init_box).view(-1, 4).to(coco_demo.device), (im.shape[1], im.shape[0]), mode="xyxy")
    proposal.add_field("labels", torch.tensor([label, label]).to(coco_demo.device))
    proposal.add_field("box_labels", torch.tensor([label, label]).to(coco_demo.device))
    proposal.add_field("scores", torch.tensor([1, 1]).to(coco_demo.device))

    proposal = [proposal]

    # compute predictions
    with torch.no_grad():
        features = coco_demo.model.backbone(image_list.tensors)

        x, detections, loss_iou = coco_demo.model.roi_heads.iou(features, proposal)

    refined_box = detections[0].bbox.cpu().view(-1).int().tolist()

    cv2.rectangle(im_disp, (init_box[0][0], init_box[0][1]), (init_box[0][2], init_box[0][3]), (0, 255, 0), 1)
    cv2.rectangle(im_disp, (refined_box[0], refined_box[1]), (refined_box[2], refined_box[3]), (0, 0, 255), 1)
    cv2.imshow("Display", im_disp)
    cv2.waitKey(0)
    # Copy image

    # Extract features

    # Run IoUNet


if __name__ == "__main__":
    main()
