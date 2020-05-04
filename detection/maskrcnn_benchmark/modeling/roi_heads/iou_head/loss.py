# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.utils import cat
import random
import math


def iou(reference, proposals):
    """Compute the IoU between a reference box with multiple proposal boxes.

    args:
        reference - Tensor of shape (1, 4).
        proposals - Tensor of shape (num_proposals, 4)

    returns:
        torch.Tensor - Tensor of shape (num_proposals,) containing IoU of reference box with each proposal box.
    """

    # Intersection box
    tl = torch.max(reference[:,:2], proposals[:,:2])
    br = torch.min(reference[:,:2] + reference[:,2:], proposals[:,:2] + proposals[:,2:])
    sz = (br - tl).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = reference[:,2:].prod(dim=1) + proposals[:,2:].prod(dim=1) - intersection

    return intersection / union


def rand_uniform(a, b, shape=1):
    """ sample numbers uniformly between a and b.
    args:
        a - lower bound
        b - upper bound
        shape - shape of the output tensor

    returns:
        torch.Tensor - tensor of shape=shape
    """
    return (b - a) * torch.rand(shape) + a


def perturb_box(box, min_iou=0.5, sigma_factor=0.1):
    """ Perturb the input box by adding gaussian noise to the co-ordinates

     args:
        box - input box
        min_iou - minimum IoU overlap between input box and the perturbed box
        sigma_factor - amount of perturbation, relative to the box size. Can be either a single element, or a list of
                        sigma_factors, in which case one of them will be uniformly sampled. Further, each of the
                        sigma_factor element can be either a float, or a tensor
                        of shape (4,) specifying the sigma_factor per co-ordinate

    returns:
        torch.Tensor - the perturbed box
    """

    if isinstance(sigma_factor, list):
        # If list, sample one sigma_factor as current sigma factor
        c_sigma_factor = random.choice(sigma_factor)
    else:
        c_sigma_factor = sigma_factor

    if not isinstance(c_sigma_factor, torch.Tensor):
        c_sigma_factor = c_sigma_factor * torch.ones(4)

    perturb_factor = torch.sqrt(box[2]*box[3])*c_sigma_factor

    # multiple tries to ensure that the perturbed box has iou > min_iou with the input box
    for i_ in range(100):
        c_x = box[0] + 0.5*box[2]
        c_y = box[1] + 0.5 * box[3]
        c_x_per = random.gauss(c_x, perturb_factor[0])
        c_y_per = random.gauss(c_y, perturb_factor[1])

        w_per = random.gauss(box[2], perturb_factor[2])
        h_per = random.gauss(box[3], perturb_factor[3])

        box_per = torch.Tensor([c_x_per - 0.5*w_per, c_y_per - 0.5*h_per, w_per, h_per])

        if box_per[2] <= 0:
            box_per[2] = box[2]*rand_uniform(0.15, 0.5)

        if box_per[3] <= 0:
            box_per[3] = box[3]*rand_uniform(0.15, 0.5)

        box_iou = iou(box.view(1, 4), box_per.view(1, 4))

        # if there is sufficient overlap, return
        if box_iou > min_iou:
            return box_per, box_iou

        # else reduce the perturb factor
        perturb_factor *= 0.9

    return box_per, box_iou


def rect_to_rel(bb, sz_norm=None):
    c = bb[...,:2] + 0.5 * bb[...,2:]
    if sz_norm is None:
        c_rel = c / bb[...,2:]
    else:
        c_rel = c / sz_norm
    sz_rel = torch.log(bb[...,2:])
    return torch.cat((c_rel, sz_rel), dim=-1)


def rel_to_rect(bb, sz_norm=None):
    sz = torch.exp(bb[...,2:])
    if sz_norm is None:
        c = bb[...,:2] * sz
    else:
        c = bb[...,:2] * sz_norm
    tl = c - 0.5 * sz
    return torch.cat((tl, sz), dim=-1)


def gauss_density_centered(x, std):
    return torch.exp(-0.5*(x / std)**2) / (math.sqrt(2*math.pi)*std)

def gmm_density_centered(x, std):
    """
    Assumes dim=-1 is the component dimension and dim=-2 is feature dimension. Rest are sample dimension.
    """
    if x.dim() == std.dim() - 1:
        x = x.unsqueeze(-1)
    elif not (x.dim() == std.dim() and x.shape[-1] == 1):
        raise ValueError('Last dimension must be the gmm stds.')
    return gauss_density_centered(x, std).prod(-2).mean(-1)


def sample_gmm_centered(std, num_samples=1):
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components

    std = std.view(1, num_dims, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0,:,k].t()

    # Sample
    x_centered = std_samp * torch.randn(num_samples, num_dims)
    prob_dens = gmm_density_centered(x_centered, std)

    return x_centered, prob_dens

def sample_gmm(mean, std, num_samples=1):
    num_dims = mean.numel()
    num_components = std.shape[-1]

    mean = mean.view(1,num_dims)
    std = std.view(1, -1, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0,:,k].t()

    # Sample
    x_centered = std_samp * torch.randn(num_samples, num_dims)
    x = x_centered + mean
    prob_dens = gmm_density_centered(x_centered, std)

    return x, prob_dens


def sample_box_gmm(mean_box, proposal_sigma, gt_sigma=None, num_samples=1, add_mean_box=False):
    center_std = torch.Tensor([s[0] for s in proposal_sigma])
    sz_std = torch.Tensor([s[1] for s in proposal_sigma])
    std = torch.stack([center_std, center_std, sz_std, sz_std])

    mean_box = mean_box.view(1,4)
    sz_norm = mean_box[:,2:].clone()

    # Sample boxes
    proposals_rel_centered, proposal_density = sample_gmm_centered(std, num_samples)

    # Add mean and map back
    mean_box_rel = rect_to_rel(mean_box, sz_norm)
    proposals_rel = proposals_rel_centered + mean_box_rel
    proposals = rel_to_rect(proposals_rel, sz_norm)

    if gt_sigma is None or gt_sigma[0] == 0 and gt_sigma[1] == 0:
        gt_density = torch.zeros_like(proposal_density)
    else:
        std_gt = torch.Tensor([gt_sigma[0], gt_sigma[0], gt_sigma[1], gt_sigma[1]]).view(1,4)
        gt_density = gauss_density_centered(proposals_rel_centered, std_gt).prod(-1)

    if add_mean_box:
        proposals = torch.cat((mean_box, proposals))
        proposal_density = torch.cat((torch.Tensor([-1]), proposal_density))
        gt_density = torch.cat((torch.Tensor([1]), gt_density))

    return proposals, proposal_density, gt_density


def kl_regression_loss(scores, sample_density, gt_density, mc_dim=0, eps=0.0, size_average=True):
    """mc_dim is dimension of MC samples."""
    L = torch.log(torch.mean(torch.exp(scores) / (sample_density + eps), dim=mc_dim)) - \
        torch.mean(scores * (gt_density / (sample_density + eps)), dim=mc_dim)

    if size_average:
        return L.mean()
    else:
        return L


def ml_regression_loss(scores, sample_density, gt_density, mc_dim=0, eps=0.0, exp_max=None, size_average=True):
    """mc_dim is dimension of MC samples."""

    assert mc_dim == 1
    assert (sample_density[:,0,...] == -1).all()
    assert (gt_density[:,0,...] == 1).all()

    exp_val = scores[:, 1:, ...] - torch.log(sample_density[:, 1:, ...] + eps)

    if exp_max is None:
        bias = 0
        bias_squeeze = 0
    else:
        bias = (torch.max(exp_val.detach(), dim=mc_dim, keepdim=True)[0] - exp_max).clamp(min=0)
        bias_squeeze = bias.squeeze(dim=mc_dim)

    L = torch.log(torch.mean(torch.exp(exp_val - bias), dim=mc_dim)) + bias_squeeze - scores[:, 0, ...]

    if size_average:
        loss = L.mean()
    else:
        loss = L
    return loss

def ml_regression_loss_logsumexp(scores, sample_density, gt_density, mc_dim=0, eps=0.0, exp_max=None, size_average=True): ##############################################
    """mc_dim is dimension of MC samples."""
    # (scores has shape: (num_bboxes_in_batch, M)) (M == 128) (num_bboxes_in_batch can be different for each batch)
    # (sample_density has shape: (num_bboxes_in_batch, M))
    # (gt_density has shape: (num_bboxes_in_batch, M))

    assert mc_dim == 1
    assert (sample_density[:,0,...] == -1).all()
    assert (gt_density[:,0,...] == 1).all()

    scores_samples = scores[:, 1:] # (shape: (num_bboxes_in_batch, M-1))
    q_y_samples = sample_density[:, 1:] # (shape: (num_bboxes_in_batch, M-1))
    scores_gt = scores[:, 0] # (shape: (num_bboxes_in_batch))
    num_samples = scores_samples.size(1) # (M-1)

    log_Z = torch.logsumexp(scores_samples - torch.log(q_y_samples), dim=1) - math.log(num_samples) # (shape: (num_bboxes_in_batch))

    L = log_Z - scores_gt # (shape: (num_bboxes_in_batch))

    # print (scores_samples.size())
    # print (q_y_samples.size())
    # print (scores_gt.size())
    # print (num_samples)
    # print (math.log(num_samples))
    # print ("{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}")

    if size_average:
        loss = L.mean()
    else:
        loss = L
    return loss

# def kl_regression_loss_logsumexp(scores, sample_density, gt_density, mc_dim=0, eps=0.0, exp_max=None, size_average=True): ##############################################
#     """mc_dim is dimension of MC samples."""
#     # (scores has shape: (num_bboxes_in_batch, M)) (M == 128) (num_bboxes_in_batch can be different for each batch)
#     # (sample_density has shape: (num_bboxes_in_batch, M))
#     # (gt_density has shape: (num_bboxes_in_batch, M))
#
#     exp_val = scores - torch.log(sample_density + eps)
#
#     L = torch.logsumexp(exp_val, dim=1) - math.log(scores.shape[1]) - torch.mean(scores * (gt_density / (sample_density + eps)), dim=1)
#
#     if size_average:
#         loss = L.mean()
#     else:
#         loss = L
#     return loss


class IoUNetLossComputation(object):
    """
    """

    def __init__(
        self,
        num_proposal=16,
        proposal_min_overlap=0.5,
        cls_agnostic_iou_pred=False,
        num_pre_generated_boxes=50000,
        sampling_type="default_iou",
        proposal_sigma=None,
        gt_sigma=None
    ):
        self.num_proposal = num_proposal
        self.proposal_min_overlap = proposal_min_overlap
        self.cls_agnostic_iou_pred = cls_agnostic_iou_pred
        self.num_pre_generated_boxes = num_pre_generated_boxes

        self.sampling_type = sampling_type

        if sampling_type == "default_iou":
            self.jittered_boxes, self.jittered_box_ious = self.generate_jittered_boxes()
            self.sample_probability = self.get_sample_probability()

        self.proposal_sigma = proposal_sigma
        self.gt_sigma = gt_sigma

    def generate_jittered_boxes(self):
        jittered_boxes = []
        jittered_box_ious = []

        base_box = torch.tensor([-0.5, -0.5, 1.0, 1.0])
        for i in range(self.num_pre_generated_boxes):
            box_per, box_iou = perturb_box(base_box.clone(), min_iou=self.proposal_min_overlap,
                                           sigma_factor= [0.004, 0.01, 0.05, 0.1, 0.2, 0.3])
            jittered_boxes.append(box_per)
            jittered_box_ious.append(box_iou)

        # TODO use bboxlist
        return torch.stack(jittered_boxes, dim=0), torch.stack(jittered_box_ious, dim=0)

    def get_sample_probability(self):
        num_bins = 100
        iou_hist = torch.histc(self.jittered_box_ious, bins=num_bins, min=self.proposal_min_overlap, max=1.0)

        weight = 1 / (iou_hist + 1)

        idx = (self.jittered_box_ious - self.proposal_min_overlap) / (1.0 - self.proposal_min_overlap)
        idx = (idx * (num_bins - 1)).long()

        sample_probability = weight[idx]
        return sample_probability.view(-1)

    def sample_jitter(self, num_samples):
        sampled_ids = torch.multinomial(self.sample_probability, num_samples, replacement=True)

        return self.jittered_boxes[sampled_ids, :], self.jittered_box_ious[sampled_ids]

    def sample_jittered_boxes(self, gt_boxes):
        """
        :param gt_boxes: BoxList containining the gt boxes for each image
        :return:
        """

        orig_mode_list = [b.mode for b in gt_boxes]
        gt_boxes = [b.convert('xywh') for b in gt_boxes]

        gt_iou_list = []
        out_boxes = []

        if self.sampling_type == "default_iou":
            for gt_b in gt_boxes:
                b = gt_b.bbox.view(-1, 4)
                labels = gt_b.get_field("labels")

                jittered_base_boxes, gt_iou = self.sample_jitter(b.shape[0] * self.num_proposal)

                jittered_base_boxes = jittered_base_boxes.to(b.device)
                gt_iou = gt_iou.to(b.device)
                jittered_base_boxes_scaled = jittered_base_boxes.view(b.shape[0], self.num_proposal, 4) * b[:, 2:].repeat(1, 2).view(-1, 1, 4)
                b_center = b[:, :2] + 0.5 * b[:, 2:]

                jittered_base_boxes_scaled[..., :2] += b_center.view(-1, 1, 2)
                labels = labels.view(-1,1).repeat(1, self.num_proposal)

                new_box = BoxList(jittered_base_boxes_scaled.view(-1, 4), image_size=gt_b.size, mode='xywh')
                new_box.add_field("labels", labels.view(-1))

                out_boxes.append(new_box)
                gt_iou_list.append(gt_iou)
        elif self.sampling_type == "default_ml":
            for gt_b in gt_boxes:
                device = gt_b.bbox.device
                b = gt_b.bbox.view(-1, 4).cpu()
                labels = gt_b.get_field("labels")

                out_boxes_list = []
                for i in range(b.shape[0]):
                    proposals, _, _ = sample_box_gmm(b[i, :], self.proposal_sigma, self.gt_sigma, self.num_proposal - 1,
                                                     True)

                    gt_iou = iou(b[i:i+1, :], proposals)
                    proposals = proposals.to(device)
                    gt_iou = gt_iou.to(device)
                    gt_iou = gt_iou.view(-1, 1)
                    out_boxes_list.append(proposals)
                    gt_iou_list.append(gt_iou)

                out_boxes_t = torch.cat(out_boxes_list, dim=0)

                new_box = BoxList(out_boxes_t, image_size=gt_b.size, mode='xywh')
                labels = labels.view(-1, 1).repeat(1, self.num_proposal)
                new_box.add_field("labels", labels.view(-1))

                out_boxes.append(new_box)
        else:
            raise ValueError

        out_boxes = [b.convert(m) for b, m in zip(out_boxes, orig_mode_list)]
        gt_boxes = [b.convert('xyxy') for b in gt_boxes]

        # TODO is this safe?
        self._gt_iou = 2 * torch.cat(gt_iou_list, dim=0) - 1
        self._proposals = out_boxes

        # TODO check output distribution
        return out_boxes

    def __call__(self, iou_score):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        device = iou_score.device

        # TODO handle multi-class stuff
        gt_iou = self._gt_iou

        labels = cat([proposal.get_field("labels") for proposal in self._proposals], dim=0).long()

        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]

        if self.cls_agnostic_iou_pred:
            map_inds = torch.tensor([0], device=device)
        else:
            map_inds = labels_pos[:, None]

        iou_loss = smooth_l1_loss(
            iou_score[sampled_pos_inds_subset[:, None], map_inds],
            gt_iou[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )

        iou_loss = iou_loss / labels.numel()
        return iou_loss


class KLIoUNetLossComputation(object):
    """
    """

    def __init__(
        self,
        num_proposal=128,
        gt_sigma=(0.125, 0.125),
        proposal_sigma=((0.125, 0.125), (0.25, 0.25), (0.5, 0.5), (1.0, 1.0)),
        cls_agnostic_iou_pred=False,
    ):
        self.num_proposal = num_proposal
        self.gt_sigma = gt_sigma
        self.proposal_sigma = proposal_sigma
        self.cls_agnostic_iou_pred = cls_agnostic_iou_pred

    def sample_jittered_boxes(self, gt_boxes):
        """
        :param gt_boxes: BoxList containining the gt boxes for each image
        :return:
        """
        orig_mode_list = [b.mode for b in gt_boxes]
        gt_boxes = [b.convert('xywh') for b in gt_boxes]

        proposal_density_list = []
        gt_density_list = []

        jittered_boxes = []
        for gt_b in gt_boxes:
            device = gt_b.bbox.device
            b = gt_b.bbox.view(-1, 4).cpu()
            labels = gt_b.get_field("labels")

            out_boxes_list = []

            for i in range(b.shape[0]):
                proposals, proposal_density, gt_density = sample_box_gmm(b[i, :], self.proposal_sigma, self.gt_sigma,
                                                                         self.num_proposal, False)

                proposals = proposals.to(device)
                proposal_density = proposal_density.to(device)
                gt_density = gt_density.to(device)

                out_boxes_list.append(proposals)
                proposal_density_list.append(proposal_density)
                gt_density_list.append(gt_density)

            out_boxes_t = torch.cat(out_boxes_list, dim=0)

            new_box = BoxList(out_boxes_t, image_size=gt_b.size, mode='xywh')
            labels = labels.view(-1, 1).repeat(1, self.num_proposal)
            new_box.add_field("labels", labels.view(-1))

            jittered_boxes.append(new_box)

        jittered_boxes = [b.convert(m) for b, m in zip(jittered_boxes, orig_mode_list)]

        # TODO is this safe?
        self._proposal_density = torch.cat(proposal_density_list, dim=0)
        self._gt_density = torch.cat(gt_density_list, dim=0)
        self._proposals = jittered_boxes

        # TODO check output distribution
        return jittered_boxes

    def __call__(self, score):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        device = score.device

        # TODO handle multi-class stuff
        gt_density = self._gt_density
        proposal_density = self._proposal_density

        labels = cat([proposal.get_field("labels") for proposal in self._proposals], dim=0).long()

        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]

        if self.cls_agnostic_iou_pred:
            map_inds = torch.tensor([0], device=device)
        else:
            map_inds = labels_pos[:, None]

        iou_loss = kl_regression_loss(
            score[sampled_pos_inds_subset[:, None], map_inds].view(-1, self.num_proposal),
            proposal_density[sampled_pos_inds_subset].view(-1, self.num_proposal),
            gt_density[sampled_pos_inds_subset].view(-1, self.num_proposal),
            size_average=False,
            mc_dim=1)

        iou_loss = iou_loss.sum() / (labels.numel() / self.num_proposal)
        return iou_loss


class MLIoUNetLossComputation(object):
    """
    """

    def __init__(
        self,
        num_proposal=128,
        gt_sigma=(0.125, 0.125),
        proposal_sigma=((0.125, 0.125), (0.25, 0.25), (0.5, 0.5), (1.0, 1.0)),
        cls_agnostic_iou_pred=False,
        exp_clamp=None
    ):
        self.num_proposal = num_proposal
        self.gt_sigma = gt_sigma
        self.proposal_sigma = proposal_sigma
        self.cls_agnostic_iou_pred = cls_agnostic_iou_pred
        self.exp_clamp = exp_clamp

    def sample_jittered_boxes(self, gt_boxes):
        """
        :param gt_boxes: BoxList containining the gt boxes for each image
        :return:
        """
        orig_mode_list = [b.mode for b in gt_boxes]
        gt_boxes = [b.convert('xywh') for b in gt_boxes]

        proposal_density_list = []
        gt_density_list = []

        jittered_boxes = []
        for gt_b in gt_boxes:
            device = gt_b.bbox.device
            b = gt_b.bbox.view(-1, 4).cpu()
            labels = gt_b.get_field("labels")

            out_boxes_list = []

            for i in range(b.shape[0]):
                proposals, proposal_density, gt_density = sample_box_gmm(b[i, :], self.proposal_sigma, self.gt_sigma,
                                                                         self.num_proposal-1, True)

                proposals = proposals.to(device)
                proposal_density = proposal_density.to(device)
                gt_density = gt_density.to(device)

                out_boxes_list.append(proposals)
                proposal_density_list.append(proposal_density)
                gt_density_list.append(gt_density)

            out_boxes_t = torch.cat(out_boxes_list, dim=0)

            new_box = BoxList(out_boxes_t, image_size=gt_b.size, mode='xywh')
            labels = labels.view(-1, 1).repeat(1, self.num_proposal)
            new_box.add_field("labels", labels.view(-1))

            jittered_boxes.append(new_box)

        jittered_boxes = [b.convert(m) for b, m in zip(jittered_boxes, orig_mode_list)]

        # TODO is this safe?
        self._proposal_density = torch.cat(proposal_density_list, dim=0)
        self._gt_density = torch.cat(gt_density_list, dim=0)
        self._proposals = jittered_boxes

        # TODO check output distribution
        return jittered_boxes

    def __call__(self, score):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        device = score.device

        # TODO handle multi-class stuff
        gt_density = self._gt_density
        proposal_density = self._proposal_density

        labels = cat([proposal.get_field("labels") for proposal in self._proposals], dim=0).long()

        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]

        if self.cls_agnostic_iou_pred:
            map_inds = torch.tensor([0], device=device)
        else:
            map_inds = labels_pos[:, None]

        iou_loss = ml_regression_loss(
            score[sampled_pos_inds_subset[:, None], map_inds].view(-1, self.num_proposal),
            proposal_density[sampled_pos_inds_subset].view(-1, self.num_proposal),
            gt_density[sampled_pos_inds_subset].view(-1, self.num_proposal),
            exp_max=10,
            size_average=False,
            mc_dim=1)

        iou_loss = iou_loss.sum() / (labels.numel() / self.num_proposal)
        return iou_loss

class LossComputation_mlis(object): #########################################################################
    """
    """

    def __init__(
        self,
        num_proposal=128,
        gt_sigma=(0.125, 0.125),
        proposal_sigma=((0.125, 0.125), (0.25, 0.25), (0.5, 0.5), (1.0, 1.0)),
        cls_agnostic_iou_pred=False,
        exp_clamp=None
    ):
        self.num_proposal = num_proposal
        self.gt_sigma = gt_sigma
        self.proposal_sigma = proposal_sigma
        self.cls_agnostic_iou_pred = cls_agnostic_iou_pred
        self.exp_clamp = exp_clamp

    def sample_jittered_boxes(self, gt_boxes): ######################################################
        """
        :param gt_boxes: BoxList containining the gt boxes for each image
        :return:
        """
        orig_mode_list = [b.mode for b in gt_boxes]
        gt_boxes = [b.convert('xywh') for b in gt_boxes]

        proposal_density_list = []
        gt_density_list = []

        jittered_boxes = []
        for gt_b in gt_boxes:
            device = gt_b.bbox.device
            b = gt_b.bbox.view(-1, 4).cpu()
            labels = gt_b.get_field("labels")

            out_boxes_list = []

            for i in range(b.shape[0]):
                proposals, proposal_density, gt_density = sample_box_gmm(b[i, :], self.proposal_sigma, self.gt_sigma,
                                                                         self.num_proposal-1, True)

                proposals = proposals.to(device)
                proposal_density = proposal_density.to(device)
                gt_density = gt_density.to(device)

                out_boxes_list.append(proposals)
                proposal_density_list.append(proposal_density)
                gt_density_list.append(gt_density)

            out_boxes_t = torch.cat(out_boxes_list, dim=0)

            new_box = BoxList(out_boxes_t, image_size=gt_b.size, mode='xywh')
            labels = labels.view(-1, 1).repeat(1, self.num_proposal)
            new_box.add_field("labels", labels.view(-1))

            jittered_boxes.append(new_box)

        jittered_boxes = [b.convert(m) for b, m in zip(jittered_boxes, orig_mode_list)]

        # TODO is this safe?
        self._proposal_density = torch.cat(proposal_density_list, dim=0)
        self._gt_density = torch.cat(gt_density_list, dim=0)
        self._proposals = jittered_boxes

        # TODO check output distribution
        return jittered_boxes

    def __call__(self, score): ##############################################################
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        # (score has shape: (num_bboxes, 81)) (num_bboxes can be different for every batch, e.g. 13952, 20608, ...)

        device = score.device

        # TODO handle multi-class stuff
        gt_density = self._gt_density # (shape: (num_bboxes))
        proposal_density = self._proposal_density # (shape: (num_bboxes))

        labels = cat([proposal.get_field("labels") for proposal in self._proposals], dim=0).long()

        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]

        if self.cls_agnostic_iou_pred:
            map_inds = torch.tensor([0], device=device)
        else:
            map_inds = labels_pos[:, None]

        # (score[sampled_pos_inds_subset[:, None], map_inds].view(-1, self.num_proposal) has shape: (num_gt_bboxes_in_batch, M)) (M == 128) (num_bboxes_in_batch can be different for each batch, e.g. 151, 109, ...)
        # (num_gt_bboxes_in_batch == num_bboxes/M) (score has shape: (num_bboxes, 81))
        # (proposal_density[sampled_pos_inds_subset].view(-1, self.num_proposal) has shape: (num_gt-bboxes_in_batch, M))
        # (gt_density[sampled_pos_inds_subset].view(-1, self.num_proposal) has shape: (num_gt_bboxes_in_batch, M))

        iou_loss = ml_regression_loss_logsumexp(
            score[sampled_pos_inds_subset[:, None], map_inds].view(-1, self.num_proposal),
            proposal_density[sampled_pos_inds_subset].view(-1, self.num_proposal),
            gt_density[sampled_pos_inds_subset].view(-1, self.num_proposal),
            exp_max=10,
            size_average=False,
            mc_dim=1)
        # (iou_loss has shape: (num_bboxes_in_batch)) (num_bboxes_in_batch can be different for each batch, e.g. 49, 68, 142, 127, ...)

        iou_loss = iou_loss.sum() / (labels.numel() / self.num_proposal)

        # (labels.numel() / self.num_proposal == num_bboxes_in_batch)
        # (so, could also just have done iou_loss = torch.mean(iou_loss)?)

        return iou_loss

# class LossComputation_kldis(object): #########################################################################
#     """
#     """
#
#     def __init__(
#         self,
#         num_proposal=128,
#         gt_sigma=(0.125, 0.125),
#         proposal_sigma=((0.125, 0.125), (0.25, 0.25), (0.5, 0.5), (1.0, 1.0)),
#         cls_agnostic_iou_pred=False,
#         exp_clamp=None
#     ):
#         self.num_proposal = num_proposal
#         self.gt_sigma = gt_sigma
#         self.proposal_sigma = proposal_sigma
#         self.cls_agnostic_iou_pred = cls_agnostic_iou_pred
#         self.exp_clamp = exp_clamp
#
#     def sample_jittered_boxes(self, gt_boxes): ######################################################
#         """
#         :param gt_boxes: BoxList containining the gt boxes for each image
#         :return:
#         """
#         orig_mode_list = [b.mode for b in gt_boxes]
#         gt_boxes = [b.convert('xywh') for b in gt_boxes]
#
#         proposal_density_list = []
#         gt_density_list = []
#
#         jittered_boxes = []
#         for gt_b in gt_boxes:
#             device = gt_b.bbox.device
#             b = gt_b.bbox.view(-1, 4).cpu()
#             labels = gt_b.get_field("labels")
#
#             out_boxes_list = []
#
#             for i in range(b.shape[0]):
#                 proposals, proposal_density, gt_density = sample_box_gmm(b[i, :], self.proposal_sigma, self.gt_sigma,
#                                                                          self.num_proposal, False)
#
#                 proposals = proposals.to(device)
#                 proposal_density = proposal_density.to(device)
#                 gt_density = gt_density.to(device)
#
#                 out_boxes_list.append(proposals)
#                 proposal_density_list.append(proposal_density)
#                 gt_density_list.append(gt_density)
#
#             out_boxes_t = torch.cat(out_boxes_list, dim=0)
#
#             new_box = BoxList(out_boxes_t, image_size=gt_b.size, mode='xywh')
#             labels = labels.view(-1, 1).repeat(1, self.num_proposal)
#             new_box.add_field("labels", labels.view(-1))
#
#             jittered_boxes.append(new_box)
#
#         jittered_boxes = [b.convert(m) for b, m in zip(jittered_boxes, orig_mode_list)]
#
#         # TODO is this safe?
#         self._proposal_density = torch.cat(proposal_density_list, dim=0)
#         self._gt_density = torch.cat(gt_density_list, dim=0)
#         self._proposals = jittered_boxes
#
#         # TODO check output distribution
#         return jittered_boxes
#
#     def __call__(self, score): ##############################################################
#         """
#         Computes the loss for Faster R-CNN.
#         This requires that the subsample method has been called beforehand.
#
#         Arguments:
#             class_logits (list[Tensor])
#             box_regression (list[Tensor])
#
#         Returns:
#             classification_loss (Tensor)
#             box_loss (Tensor)
#         """
#
#         # (score has shape: (num_bboxes, 81)) (num_bboxes can be different for every batch, e.g. 13952, 20608, ...)
#
#         device = score.device
#
#         # TODO handle multi-class stuff
#         gt_density = self._gt_density # (shape: (num_bboxes))
#         proposal_density = self._proposal_density # (shape: (num_bboxes))
#
#         labels = cat([proposal.get_field("labels") for proposal in self._proposals], dim=0).long()
#
#         sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
#         labels_pos = labels[sampled_pos_inds_subset]
#
#         if self.cls_agnostic_iou_pred:
#             map_inds = torch.tensor([0], device=device)
#         else:
#             map_inds = labels_pos[:, None]
#
#         # (score[sampled_pos_inds_subset[:, None], map_inds].view(-1, self.num_proposal) has shape: (num_gt_bboxes_in_batch, M)) (M == 128) (num_bboxes_in_batch can be different for each batch, e.g. 151, 109, ...)
#         # (num_gt_bboxes_in_batch == num_bboxes/M) (score has shape: (num_bboxes, 81))
#         # (proposal_density[sampled_pos_inds_subset].view(-1, self.num_proposal) has shape: (num_gt-bboxes_in_batch, M))
#         # (gt_density[sampled_pos_inds_subset].view(-1, self.num_proposal) has shape: (num_gt_bboxes_in_batch, M))
#
#         iou_loss = kl_regression_loss_logsumexp(
#             score[sampled_pos_inds_subset[:, None], map_inds].view(-1, self.num_proposal),
#             proposal_density[sampled_pos_inds_subset].view(-1, self.num_proposal),
#             gt_density[sampled_pos_inds_subset].view(-1, self.num_proposal),
#             exp_max=10,
#             size_average=False,
#             mc_dim=1)
#         # (iou_loss has shape: (num_bboxes_in_batch)) (num_bboxes_in_batch can be different for each batch, e.g. 49, 68, 142, 127, ...)
#
#         iou_loss = iou_loss.sum() / (labels.numel() / self.num_proposal)
#
#         # (labels.numel() / self.num_proposal == num_bboxes_in_batch)
#         # (so, could also just have done iou_loss = torch.mean(iou_loss)?)
#
#         return iou_loss


def make_roi_iou_loss_evaluator(cfg):
    num_proposal = cfg.MODEL.ROI_IOU_HEAD.NUM_TRAIN_PROPOSALS
    proposal_min_overlap = cfg.MODEL.ROI_IOU_HEAD.MIN_OVERLAP_PROPOSAL
    cls_agnostic_iou_pred = cfg.MODEL.CLS_AGNOSTIC_IOU_PRED
    loss_type = cfg.MODEL.ROI_IOU_HEAD.LOSS_TYPE

    sampling_type = cfg.MODEL.ROI_IOU_HEAD.PROPOSAL_SAMPLING_TYPE
    proposal_sigma = cfg.MODEL.ROI_IOU_HEAD.PROPOSAL_SIGMA
    gt_sigma = cfg.MODEL.ROI_IOU_HEAD.GT_SIGMA

    if loss_type == "ML-IS":
        loss_evaluator = LossComputation_mlis(num_proposal=num_proposal,
                                              gt_sigma=gt_sigma,
                                              proposal_sigma=proposal_sigma,
                                              cls_agnostic_iou_pred=cls_agnostic_iou_pred,
                                              exp_clamp=None
                                              )
    elif loss_type == "KLD-IS":
        loss_evaluator = LossComputation_kldis(num_proposal=num_proposal,
                                              gt_sigma=gt_sigma,
                                              proposal_sigma=proposal_sigma,
                                              cls_agnostic_iou_pred=cls_agnostic_iou_pred,
                                              exp_clamp=None
                                              )
    else:
        raise ValueError

    return loss_evaluator
