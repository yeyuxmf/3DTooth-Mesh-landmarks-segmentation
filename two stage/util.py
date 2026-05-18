#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: util
@Time: 4/5/19 3:47 PM
"""


import numpy as np
import torch
import torch.nn.functional as F
from metrics import cal_score, reformat_scores
from config import config as cfg

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
def accumulate_net(model1, model2, decay):
    """
        operation: model1 = model1 * decay + model2 * (1 - decay)
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(
            other=par2[k].data.to(par1[k].data.device),
            alpha=1 - decay)

    par1 = dict(model1.named_buffers())
    par2 = dict(model2.named_buffers())
    for k in par1.keys():
        if par1[k].data.is_floating_point():
            par1[k].data.mul_(decay).add_(
                other=par2[k].data.to(par1[k].data.device),
                alpha=1 - decay)
        else:
            par1[k].data = par2[k].data.to(par1[k].data.device)

def get_fixed_cd(prelandmarks, fl_mask, conf_thresh):

    mask = fl_mask.squeeze()
    indices = torch.nonzero(mask.squeeze() > 0).squeeze()
    TN = indices.shape[0]

    plandmarks = [prelandmarks[ind].reshape(-1, 7) for ind in indices]

    plandmarks = torch.cat(plandmarks, dim=0)

    precls = plandmarks[..., 6].sigmoid()
    plandmarks = plandmarks[..., :3]

    pro = precls.reshape(TN, 5).detach().cpu().numpy()
    pcoord = plandmarks.reshape(TN, 5, 3).detach().cpu().numpy()

    pre_mask = torch.ge(precls, conf_thresh)  #
    plandmarks = plandmarks[pre_mask].detach().cpu().numpy()
    pre_mask = pre_mask.reshape(TN, 5).detach().cpu().numpy()



    return plandmarks, pro, pcoord


def get_match_cd(pre_matchLandmarks, match_mask, conf_thresh):

    mask = match_mask.squeeze()
    indices = torch.nonzero(mask.squeeze() > 0).squeeze()
    TN = indices.shape[0]

    matchLandmarks = [pre_matchLandmarks[ind].reshape(-1, 7) for ind in indices]

    matchLandmarks = torch.cat(matchLandmarks, dim=0)

    precls = matchLandmarks[..., 6].sigmoid()
    pred_sigmas = matchLandmarks[..., 3:6].sigmoid()
    prelandmarks = matchLandmarks[..., :3]


    pro = precls.detach().cpu().numpy()
    pcoord = prelandmarks.detach().cpu().numpy()


    pre_mask = torch.ge(precls, conf_thresh)  #
    prelandmarks = prelandmarks[pre_mask].detach().cpu().numpy()
    pre_mask = pre_mask.reshape(TN, -1).detach().cpu().numpy()


    return prelandmarks, pro, pcoord


def decoder_land(prelandmarks, matchLandmarks, fl_mask, match_mask, pred_all, nums, conf_thresh):

    fpt_land, fpro, fpcoord = get_fixed_cd(prelandmarks, fl_mask, conf_thresh)
    ######################################################
    mpt_land, mpro, mpcoord = get_match_cd(matchLandmarks, match_mask, conf_thresh)

    #######################################################################
    pred_land = np.concatenate([fpt_land, mpt_land], axis=0)

    score = [fpro[:, pi] for pi in range(len(cfg.fxiedPorder))]
    kp = [fpcoord[:, pi, :] for pi in range(len(cfg.fxiedPorder))]
    for pi in range(len(cfg.fxiedPorder)):
        pred_all[cfg.fxiedPorder[pi]][nums] = [tuple(x) for x in zip(kp[pi], score[pi])]
    pred_all["Cusp"][nums] = [tuple(x) for x in zip(mpcoord, mpro)]

    return pred_all,  pred_land

def cal_acc(model, test_loader, conf_thresh=0.3):

    gt_all =  { key:{} for key in (cfg.fxiedPorder + cfg.matchPorder)}
    pred_all = { key:{} for key in (cfg.fxiedPorder + cfg.matchPorder)}
    pred_alld = { key:{} for key in (cfg.fxiedPorder + cfg.matchPorder)}
    # pred_all: map of {classname: {meshname: [(kp, score)]}}
    # gt_all: map of {classname: {meshname: [kp]}}

    fm_cd, fm_cdd = [], []
    nums = 0
    for batch_data, label_mask, label_landmarks in test_loader:
        nums = nums + 1
        batch_data = batch_data.cuda().float()
        fl_mask = label_mask[..., 0].cuda().long()

        match_mask = label_mask[..., 1].cuda().long()
        with torch.no_grad():

            prelandmarks, matchLandmarks,heat_out, flandmarks, mLandmarks= model(batch_data, fl_mask)
            prelandmarks, matchLandmarks = prelandmarks.squeeze(), matchLandmarks.squeeze()
            flandmarks, mLandmarks = flandmarks.squeeze(), mLandmarks.squeeze()
            ######################################################

            pred_all, pred_land = decoder_land(prelandmarks, matchLandmarks, fl_mask, match_mask, pred_all, nums, conf_thresh)
            pred_alld, pred_landd = decoder_land(flandmarks, mLandmarks, fl_mask, match_mask, pred_alld, nums,
                                               conf_thresh)

            gt_land = []
            for i , key_class in enumerate(label_landmarks):
                key_point = label_landmarks[key_class].reshape(-1, 3).numpy()
                label_landmarks[key_class] = key_point
                gt_land.append(key_point)
            gt_land = np.concatenate(gt_land, axis=0)


            cdv = chamfer_distance(gt_land, pred_land)
            fm_cd.append(cdv)
            cdvd = chamfer_distance(gt_land, pred_landd)
            fm_cdd.append(cdvd)

            for pi in range(len(cfg.fxiedPorder)):
                gt_all[cfg.fxiedPorder[pi]][nums] = label_landmarks[cfg.fxiedPorder[pi]]
            gt_all["Cusp"][nums] = label_landmarks["Cusp"]
            #######################################################################

    all_metrics = cal_score(gt_all, pred_all)
    all_metricsd = cal_score(gt_all, pred_alld)
    #################################
    fm_cdv = np.mean(np.array(fm_cd))
    fm_cdvd = np.mean(np.array(fm_cdd))

    scores = reformat_scores(all_metrics)
    scoresd = reformat_scores(all_metricsd)

    return fm_cdv, all_metrics, scores, fm_cdvd, all_metricsd, scoresd




def cal_acc_seg_land(model, test_loader, conf_thresh=0.3):

    gt_all =  { key:{} for key in (cfg.fxiedPorder + cfg.matchPorder)}
    pred_all = { key:{} for key in (cfg.fxiedPorder + cfg.matchPorder)}
    pred_alld = { key:{} for key in (cfg.fxiedPorder + cfg.matchPorder)}
    # pred_all: map of {classname: {meshname: [(kp, score)]}}
    # gt_all: map of {classname: {meshname: [kp]}}

    fm_cd, fm_cdd = [], []
    nums = 0
    for batch_data, label_mask, label_landmarks in test_loader:
        nums = nums + 1
        batch_data = batch_data.cuda().float()
        fl_mask = label_mask[..., 0].cuda().long()

        match_mask = label_mask[..., 1].cuda().long()
        with torch.no_grad():

            prelandmarks, matchLandmarks,heat_out, flandmarks, mLandmarks, seg_score= model(batch_data, fl_mask)
            prelandmarks, matchLandmarks = prelandmarks.squeeze(), matchLandmarks.squeeze()
            flandmarks, mLandmarks = flandmarks[-1].squeeze(), mLandmarks[-1].squeeze()
            ######################################################

            pred_all, pred_land = decoder_land(prelandmarks, matchLandmarks, fl_mask, match_mask, pred_all, nums, conf_thresh)
            pred_alld, pred_landd = decoder_land(flandmarks, mLandmarks, fl_mask, match_mask, pred_alld, nums,conf_thresh)

            gt_land = []
            for i , key_class in enumerate(label_landmarks):
                key_point = label_landmarks[key_class].reshape(-1, 3).numpy()
                label_landmarks[key_class] = key_point
                gt_land.append(key_point)
            gt_land = np.concatenate(gt_land, axis=0)


            cdv = chamfer_distance(gt_land, pred_land)
            fm_cd.append(cdv)
            cdvd = chamfer_distance(gt_land, pred_landd)
            fm_cdd.append(cdvd)

            for pi in range(len(cfg.fxiedPorder)):
                gt_all[cfg.fxiedPorder[pi]][nums] = label_landmarks[cfg.fxiedPorder[pi]]
            gt_all["Cusp"][nums] = label_landmarks["Cusp"]
            #######################################################################

    all_metrics = cal_score(gt_all, pred_all)
    all_metricsd = cal_score(gt_all, pred_alld)
    #################################
    fm_cdv = np.mean(np.array(fm_cd))
    fm_cdvd = np.mean(np.array(fm_cdd))

    scores = reformat_scores(all_metrics)
    scoresd = reformat_scores(all_metricsd)

    return fm_cdv, all_metrics, scores, fm_cdvd, all_metricsd, scoresd



def chamfer_distance(points1, points2):
    # 计算每个点到另一组点云的最近邻距离
    dist1 = np.sqrt(np.sum((points1[:, None, :] - points2[None, :, :]) ** 2, axis=-1))
    dist2 = np.sqrt(np.sum((points2[:, None, :] - points1[None, :, :]) ** 2, axis=-1))

    # 计算每个点到另一组点云的最近邻距离之和
    cd = np.sum(np.min(dist1, axis=1)) + np.sum(np.min(dist2, axis=1))

    # 计算平均 Chamfer Distance
    avg_cd = cd / (points1.shape[0] + points2.shape[0])

    return avg_cd

