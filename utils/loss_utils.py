import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size(1)
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes


def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1).mean(dim=0)
    qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp


def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return d


def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):

    # find r_adv

    d = torch.rand(ul_x.shape).normal_().cuda()
    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        d.requires_grad_()
        y_hat = model(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = _l2_normalize(d.grad)
        model.zero_grad()

    r_adv = eps * d
    y_hat = model(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)\


class MultiSimilarityLoss(nn.Module):
    def __init__(self, delta):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1
        self.delta = delta
        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def _get_pos_neg_mask(self, labels):
        indices_equal = np.cast[np.bool](np.eye(np.shape(labels)[0], dtype=np.int32))
        indices_not_equal = np.logical_not(indices_equal)
        pos_mask = np.equal(np.expand_dims(labels, 0), np.expand_dims(labels, 1))
        neg_mask = ~pos_mask
        pos_mask = np.logical_and(pos_mask, indices_not_equal)

        return pos_mask, neg_mask

    def _ml_loss_pos(self, sim_mat, mask):
        exp_ = torch.exp(-self.scale_pos * (sim_mat - self.thresh))
        loss = torch.sum(torch.mul(exp_, mask), 1)
        loss = 1.0 / self.scale_pos * torch.log(1 + loss)
        return loss

    def _ml_loss_neg(self, sim_mat, mask):
        exp_ = torch.exp(self.scale_neg * (sim_mat - self.thresh))
        loss = torch.sum(torch.mul(exp_, mask), 1)
        loss = 1.0 / self.scale_neg * torch.log(1 + loss)
        return loss

    def _loss_mask(self, feats, labels, probability_v):
        probability_v = probability_v.view(-1, 1)
        pos_mask, neg_mask = self._get_pos_neg_mask(labels)
        sim_mat = torch.matmul(feats, torch.t(feats))
        probability_m = torch.matmul(probability_v, torch.t(probability_v))

        probability_m = torch.where(probability_m > self.delta, probability_m, torch.zeros_like(probability_m))
        # probability_m = torch.where(probability_m < 0.8, probability_m, torch.zeros_like(probability_m))
        # probability_m = (1. - probability_m)

        mask_neg = torch.Tensor(np.cast[np.float](neg_mask)).cuda()
        mask_pos = torch.Tensor(np.cast[np.float](pos_mask)).cuda()
        neg_pair = torch.mul(mask_neg, sim_mat)
        pos_pair = torch.mul(mask_pos, sim_mat)

        ones = torch.ones_like(pos_pair).cuda()
        anchor_negative_dist = pos_pair + torch.mul(ones, (1.0 - mask_pos))

        pos_min, _ = torch.min(anchor_negative_dist, 1)
        neg_max, _ = torch.max(neg_pair, 1)
        mask_n = (sim_mat + self.margin > pos_min.view(-1, 1)).float().cuda()
        mask_p = (sim_mat - self.margin < neg_max.view(-1, 1)).float().cuda()

        mask_neg = torch.mul(mask_neg, mask_n)
        mask_pos = torch.mul(mask_pos, mask_p)

        w_mask_neg = torch.mul(mask_neg, probability_m)
        w_mask_pos = torch.mul(mask_pos, probability_m)

        pos_loss = self._ml_loss_pos(sim_mat, w_mask_pos)
        neg_loss = self._ml_loss_neg(sim_mat, w_mask_neg)

        loss = torch.mean(pos_loss + neg_loss)

        return loss

    def forward(self, feats, labels, probability_v):

        loss = self._loss_mask(feats, labels, probability_v)

        return loss