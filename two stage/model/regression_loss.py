# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.realnvp import RealNVP
from config import config as cfg
class RLELoss(nn.Module):
    """RLE Loss.

    `Human Pose Regression With Residual Log-Likelihood Estimation
    arXiv: <https://arxiv.org/abs/2107.11291>`_.

    Code is modified from `the official implementation
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
        residual (bool): Option to add L1 loss and let the flow
            learn the residual error distribution.
        q_dis (string): Option for the identity Q(error) distribution,
            Options: "laplace" or "gaussian"
    """

    def __init__(self,
                 use_target_weight=False,
                 size_average=True,
                 residual=True,
                 q_dis='laplace',
                 in_channels=3,
                 ):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.q_dis = q_dis
        self.flow_model = RealNVP(in_channels=in_channels)

    def forward(self, output, target, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D*2]): Output regression,
                    including coords and sigmas.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        pred = output[:, :, :3]
        sigma = output[:, :, 3:6].sigmoid()

        error = (pred - target) / (sigma + 1e-9)
        # (B, K, 2)
        log_phi = self.flow_model.log_prob(error.reshape(-1, 3))
        log_phi = log_phi.reshape(target.shape[0], target.shape[1], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], target.shape[1],
                                             3)
        nf_loss = log_sigma - log_phi

        if self.residual:
            assert self.q_dis in ['laplace', 'gaussian', 'strict']
            if self.q_dis == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(
                    sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            loss = nf_loss + loss_q
        else:
            loss = nf_loss

        # if self.enable_3d_rle:
        if self.use_target_weight:
            assert target_weight is not None
            loss *= target_weight

        if self.size_average:
            loss = loss.mean()

        return loss.mean()
# ------------------------------------------------------------------------------

class KLDiscretLoss(nn.Module):
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction='none', log_target=False)

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim=1)
        return loss

    def forward(self, output_xy, target_xy):

        output_x, output_y = output_xy[..., 0], output_xy[..., 1]
        target_x, target_y = target_xy[..., 0], target_xy[..., 1]

        num_joints = output_x.size(1)
        loss = 0
        B, _, bin = output_x.shape
        coord_x_pred = output_x.reshape(-1, bin)
        coord_y_pred = output_y.reshape(-1, bin)
        coord_x_gt = target_x.reshape(-1, bin)
        coord_y_gt = target_y.reshape(-1, bin)
        loss += (self.criterion(coord_x_pred, coord_x_gt).mean())
        loss += (self.criterion(coord_y_pred, coord_y_gt).mean())

        # for idx in range(num_joints):
        #     coord_x_pred = output_x[:, idx].squeeze()
        #     coord_y_pred = output_y[:, idx].squeeze()
        #     coord_x_gt = target_x[:, idx].squeeze()
        #     coord_y_gt = target_y[:, idx].squeeze()
        #     loss += (self.criterion(coord_x_pred, coord_x_gt).mean())
        #     loss += (self.criterion(coord_y_pred, coord_y_gt).mean())
        #     loss=loss / num_joints

        return loss


class NMTNORMCritierion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(NMTNORMCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]

        if label_smoothing > 0:
            self.criterion_ = nn.KLDivLoss(reduction='none')
        else:
            self.criterion_ = nn.NLLLoss(reduction='none', ignore_index=100000)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)

        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = torch.mean(self.criterion_(scores, gtruth), dim=1)
        return loss

    def forward(self, output_x, output_y, target, target_weight):
        batch_size = output_x.size(0)
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx].squeeze()
            coord_y_pred = output_y[:, idx].squeeze()
            coord_gt = target[:, idx].squeeze()
            weight = target_weight[:, idx].squeeze()

            loss += self.criterion(coord_x_pred, coord_gt[:, 0]).mul(weight).mean()
            loss += self.criterion(coord_y_pred, coord_gt[:, 1]).mul(weight).mean()
        return loss / num_joints


class NMTCritierion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # [B,LOGITS]

        if label_smoothing > 0:
            self.criterion_ = nn.KLDivLoss(reduction='none')
        else:
            self.criterion_ = nn.NLLLoss(reduction='none', ignore_index=100000)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)

        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = torch.sum(self.criterion_(scores, gtruth), dim=1)
        return loss

    def forward(self, output_x, output_y, target, target_weight):
        batch_size = output_x.size(0)
        num_joints = output_x.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx].squeeze()
            coord_y_pred = output_y[:, idx].squeeze()
            coord_gt = target[:, idx].squeeze()
            weight = target_weight[:, idx].squeeze()
            loss += self.criterion(coord_x_pred, coord_gt[:, 0]).mul(weight).sum()
            loss += self.criterion(coord_y_pred, coord_gt[:, 1]).mul(weight).sum()
        return loss / batch_size


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)

class WingLoss(nn.Module):
    def __init__(self, omega=0.01, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_2 = (y - y_hat).pow(2).sum(dim=-1, keepdim=False)
        # delta = delta_2.sqrt()
        delta = delta_2.clamp(min=1e-6).sqrt()

        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        return loss.mean()


def focal_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    aa = torch.sum(pos_inds)

    pos_loss = (torch.log(torch.clip(pred, 1.0 * 1e-5)) * torch.pow(1 - pred, 2) * pos_inds).sum()
    neg_loss = (torch.log(torch.clip(1 - pred, 1.0 * 1e-5))* torch.pow(pred, 2) * neg_weights * neg_inds).sum()

    num_pos = pos_inds.float().sum()
    num_neg = pos_inds.float().sum()
    # pos_loss = pos_loss.sum() / 3.0
    # neg_loss = neg_loss.mean() * 19

    if num_neg == 0:
        loss = loss - pos_loss / num_pos
    else:
        loss = loss - (pos_loss/num_pos + neg_loss/num_neg) # / num_pos
    return loss



def edge_loss(pred, gt):

    relation = cfg.Relation -1
    pred1 = pred[:, relation[:,0], :]
    pred2 = pred[:, relation[:,1], :]

    gt1 = gt[:, relation[:,0], :]
    gt2 = gt[:, relation[:,1], :]

    pred_edge = pred1 - pred2
    gt_edge = gt1 - gt2

    loss = F.l1_loss(pred_edge, gt_edge)

    return loss *1e-3