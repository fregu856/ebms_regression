import torch


def nms_iou(boxes, conf, iou_scores, thresh):
    if boxes.shape[0] == 0:
        return boxes, conf, iou_scores
    conf = conf.view(-1)
    iou_scores = iou_scores.view(-1)
    boxes = boxes.view(-1, 4)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = iou_scores.argsort(descending=True)

    out_boxes = []
    out_conf = []
    out_iou_scores = []
    while order.numel() > 0:
        order = order.view(-1)
        i = order[0].item()

        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.max(torch.zeros(1).to(xx1.device), xx2 - xx1 + 1)
        h = torch.max(torch.zeros(1).to(xx1.device), yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # Find new conf
        group_inds = (ovr > thresh).nonzero() + 1
        grouped_highest_score = conf[order[group_inds]].max()

        out_boxes.append(boxes[i, :])
        out_conf.append(torch.max(conf[i], grouped_highest_score))
        out_iou_scores.append(iou_scores[i])

        inds = (ovr <= thresh).nonzero() + 1
        order = order[inds.view(-1)]

    return torch.stack(out_boxes), torch.stack(out_conf), torch.stack(out_iou_scores)
