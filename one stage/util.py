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
from sklearn.cluster import DBSCAN
from metrics import cal_score, reformat_scores
from config import config as cfg
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
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


def decode_with_nms(points, heatmap, distance_map, score_thresh=0.5, nms_radius=1, K=100):
    """
    使用 3D 距离 NMS 获取牙齿关键点

    Args:
        points: (B, N, 3) 原始点云
        heatmap: (B, N, 1) 热图分数
        distance_map: (B, N, 3) 偏移量
        score_thresh: 分数阈值
        nms_radius: NMS 抑制半径 (单位通常为 mm，如 0.3mm 内只留一个点)
        K: 每个样本最多保留的关键点数量
    """
    B, N, _ = points.shape
    # 1. 计算预测的目标中心位置
    pred_centers = points + distance_map  # (B, N, 3)
    scores = heatmap.squeeze(-1)  # (B, N)

    batch_final_centers = []
    batch_final_scores = []

    for b in range(B):
        # 2. 阈值过滤，减少计算量
        cur_scores = scores[b]
        cur_centers = pred_centers[b]

        mask = cur_scores > score_thresh
        if not mask.any():
            batch_final_centers.append(torch.zeros((0, 3), device=points.device))
            batch_final_scores.append(torch.zeros((0,), device=points.device))
            continue

        valid_scores = cur_scores[mask]
        valid_centers = cur_centers[mask]

        # 3. 按照分数从大到小排序
        indices = torch.argsort(valid_scores, descending=True)
        valid_scores = valid_scores[indices]
        valid_centers = valid_centers[indices]

        # 4. 贪心 NMS 过程
        keep = []
        is_suppressed = torch.zeros(len(valid_scores), dtype=torch.bool, device=points.device)

        for i in range(len(valid_scores)):
            if is_suppressed[i]:
                continue

            keep.append(i)
            if len(keep) >= K:
                break

            # 计算当前点与后面所有点的距离
            # 用距离平方代替开方计算，提升速度
            dist_sq = torch.sum((valid_centers[i:i + 1] - valid_centers[i + 1:]) ** 2, dim=-1)

            # 如果距离小于半径的平方，则抑制该点
            suppress_mask = dist_sq < (nms_radius ** 2)
            valid_centers[suppress_mask]
            # 更新抑制状态（注意索引偏移）
            is_suppressed[i + 1:][suppress_mask] = True

        # 5. 提取最终结果
        keep = torch.tensor(keep, device=points.device)
        batch_final_centers.append(torch.cat([valid_centers[keep], valid_scores[keep].reshape(-1, 1)], dim=-1).detach().cpu().numpy())


    return batch_final_centers



def decode_with_nms_refined(points, heatmap, distance_map, score_thresh=0.5, nms_radius=1.5, K=100):
    """
    使用 3D 距离 NMS + 权重质心提炼获取牙齿关键点

    Args:
        points: (B, N, 3) 原始点云输入
        heatmap: (B, N, 1) 每个点的热图分数
        distance_map: (B, N, 3) 每个点到其对应中心点的偏移向量
        score_thresh: 分数过滤阈值
        nms_radius: 抑制半径 (单位通常为 mm)
        K: 每个样本最多保留的关键点数量
    """
    B, N, _ = points.shape
    device = points.device

    # 1. 计算预测的目标绝对中心位置: P_center = P_origin + Offset
    pred_centers = points + distance_map  # (B, N, 3)
    scores = heatmap.squeeze(-1)  # (B, N)

    batch_final_results = []
    nms_radius_sq = nms_radius ** 2  # 预计算平方，避免循环内开方

    for b in range(B):
        cur_scores = scores[b]
        cur_centers = pred_centers[b]

        # 2. 阈值过滤
        mask = cur_scores > score_thresh
        if not mask.any():
            batch_final_results.append(np.zeros((0, 4)))  # [x, y, z, score]
            continue

        valid_scores = cur_scores[mask]
        valid_centers = cur_centers[mask]

        # 3. 按照分数从大到小排序
        indices = torch.argsort(valid_scores, descending=True)
        valid_scores = valid_scores[indices]
        valid_centers = valid_centers[indices]

        # 4. 带有中心提炼的贪心 NMS
        refined_centers = []
        refined_scores = []
        is_suppressed = torch.zeros(len(valid_scores), dtype=torch.bool, device=device)

        for i in range(len(valid_scores)):
            if is_suppressed[i]:
                continue

            # 计算当前最高分点与所有（未抑制）点的距离
            # 注意：这里我们计算与全集的距离，以便找出所有邻居进行加权
            dist_sq = torch.sum((valid_centers[i:i + 1] - valid_centers) ** 2, dim=-1)

            # 找出半径内的所有邻居点
            # 只有尚未被前面簇吸收的点才参与当前簇的计算
            neighbor_mask = (dist_sq < nms_radius_sq) & (~is_suppressed)
            neighbor_mask_ = (dist_sq < nms_radius_sq*0.3) & (~is_suppressed)
            # --- 核心改进：加权质心提炼 ---
            # 提取邻域内的所有点和分数
            cluster_centers = valid_centers[neighbor_mask_]
            cluster_scores = valid_scores[neighbor_mask_].unsqueeze(-1)  # (M, 1)

            # 计算加权平均位置：Sum(Pos * Score) / Sum(Score)
            # 这样预测越准的点（分数越高），对最终位置的影响力越大
            weights = cluster_scores / (cluster_scores.sum() + 1e-6)
            refined_pos = (cluster_centers * weights).sum(dim=0)

            # 记录结果：位置用提炼后的，分数通常保留该簇的最大值
            refined_centers.append(refined_pos)
            refined_scores.append(valid_scores[i])

            # 抑制该半径内所有的点，防止被重复计算
            is_suppressed[neighbor_mask] = True

            if len(refined_centers) >= K:
                break

        if len(refined_centers) > 0:
            res_centers = torch.stack(refined_centers)
            res_scores = torch.stack(refined_scores).unsqueeze(-1)
            # 合并为 (M, 4) 矩阵 [x, y, z, score]
            batch_result = torch.cat([res_centers, res_scores], dim=-1).detach().cpu().numpy()
        else:
            batch_result = np.zeros((0, 4))

        batch_final_results.append(batch_result)

    return batch_final_results


def fast_decode_nms_refined(points, heatmap, distance_map,
                            score_thresh=0.5, nms_radius=1.5,
                            K=100, num_passes=10):
    """
    动态分段 NMS：每一轮挑选当前最高分的 10 个点进行并行抑制。

    Args:
        num_passes: 迭代轮数 (M)
        K: 最终输出的点数（通常应等于 num_passes * 10）
    """
    B, N, C = heatmap.shape
    device = points.device
    chunk_size = 10  # 每一轮处理 10 个点

    # 1. 初始准备：计算预测中心并粗筛候选池
    # 为了保证 topk 不会报错，pool_size 必须足够大
    pool_size = min(N, max(400, num_passes * chunk_size))

    # (B, C, N, 3)
    pred_centers = (points.unsqueeze(2) + distance_map).permute(0, 2, 1, 3).contiguous()
    heatmap_bc = heatmap.permute(0, 2, 1).contiguous()

    # 提取初始的高分池
    top_scores, top_idx = torch.topk(heatmap_bc, pool_size, dim=-1)
    top_centers = torch.gather(pred_centers, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, 3))

    # 2. 动态并行 NMS
    is_suppressed = torch.zeros((B, C, pool_size), dtype=torch.bool, device=device)

    # 结果容器 (B, C, K, 4)
    out_results = torch.zeros((B, C, num_passes * chunk_size, 4), device=device)
    out_mask = torch.zeros((B, C, num_passes * chunk_size), dtype=torch.bool, device=device)

    nms_radius_sq = nms_radius ** 2
    dist_sq_mat = torch.cdist(top_centers, top_centers) ** 2

    for m in range(num_passes):
        # --- 核心逻辑：动态寻找当前活着的 Top 10 ---
        # 将已经被抑制或分数过低的点设为极小值
        current_scores = torch.where((~is_suppressed) & (top_scores > score_thresh),
                                     top_scores, torch.full_like(top_scores, -1e9))

        # 挑选本轮的 10 个候选中心
        curr_topk_scores, curr_topk_idx = torch.topk(current_scores, chunk_size, dim=-1)

        # 检查本轮是否还有有效的点（最高分是否大于阈值）
        valid_pass_mask = curr_topk_scores > score_thresh  # (B, C, 10)
        if not valid_pass_mask.any():
            break

        # 获取本轮候选点的坐标 (B, C, 10, 3)
        curr_topk_centers = torch.gather(top_centers, 2, curr_topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3))

        # --- 加权质心提炼 (Refinement) ---
        # 既然是训练，我们要保证每个选出的点都经过邻域加权以提高精度
        # 计算这 10 个点与 pool 中所有点的距离
        refine_dist_sq = torch.cdist(curr_topk_centers, top_centers) ** 2

        # 内圈用于加权提炼
        inner_mask = (refine_dist_sq < nms_radius_sq * 0.3) & (~is_suppressed).unsqueeze(2)
        weight_scores = top_scores.unsqueeze(2) * inner_mask.float()
        sum_weights = weight_scores.sum(dim=-1, keepdim=True) + 1e-6
        refined_pos = (top_centers.unsqueeze(2) * weight_scores.unsqueeze(-1)).sum(dim=3) / sum_weights

        # --- 存储结果 ---
        start_idx = m * chunk_size
        end_idx = (m + 1) * chunk_size
        out_results[:, :, start_idx:end_idx, :3] = refined_pos
        out_results[:, :, start_idx:end_idx, 3] = curr_topk_scores
        out_mask[:, :, start_idx:end_idx] = valid_pass_mask

        # --- 更新抑制状态 ---
        # 外圈用于抑制
        outer_mask = (refine_dist_sq < nms_radius_sq) & valid_pass_mask.unsqueeze(-1)
        # 只要当前轮次中的 10 个点里有任何一个覆盖了 pool 中的点，该点就被抑制
        is_suppressed |= outer_mask.any(dim=2)

    return out_results, out_mask




def decode_teeth_with_dbscan(points, heatmap, distance_map,
                             score_thresh=0.3,
                             eps=1,
                             min_samples=1):
    """
    使用 DBSCAN 聚类算法从投票点中提取精确的牙齿关键点

    Args:
        points: (B, N, 3) 原始点云坐标 (PyTorch Tensor)
        heatmap: (B, N, 1) 关键点热图置信度 (0~1)
        distance_map: (B, N, 3) 预测的偏移向量 (dx, dy, dz)
        score_thresh: 过滤低质量预测点的阈值
        eps: DBSCAN 邻域半径 (单位通常为 mm，根据牙齿间距调整)
        min_samples: 形成一个簇所需的最小点数 (增加此值可减少噪点)

    Returns:
        List[Dict]: 包含每个 batch 的结果 [{ 'centers': [], 'scores': [] }, ...]
    """
    B = points.shape[0]
    # 计算所有点的预测中心 (Votes)
    votes_all = points + distance_map  # (B, N, 3)

    batch_results = []

    # 转换为 CPU 进行聚类 (DBSCAN 在 CPU 上更稳定高效)
    votes_all_np = votes_all.detach().cpu().numpy()
    heatmap_np = heatmap.detach().cpu().numpy()

    for b in range(B):
        # 1. 根据置信度筛选点
        valid_mask = heatmap_np[b].squeeze() > score_thresh
        if not np.any(valid_mask):
            batch_results.append({'centers': np.array([]), 'scores': np.array([])})
            continue

        valid_votes = votes_all_np[b][valid_mask]  # (M, 3)
        valid_scores = heatmap_np[b][valid_mask].squeeze()  # (M,)

        # 2. 执行 DBSCAN 聚类
        # eps: 点与点之间的最大距离；min_samples: 成为核心点的最小邻居数
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(valid_votes)
        labels = clustering.labels_

        unique_labels = set(labels)
        final_centers = []
        final_scores = []

        for label in unique_labels:
            if label == -1:  # 忽略 DBSCAN 标记的噪声点
                continue

            # 3. 提取当前簇的所有成员
            class_mask = (labels == label)
            cluster_votes = valid_votes[class_mask]
            cluster_scores = valid_scores[class_mask]

            # 4. 加权求精：使用热图得分作为权重，计算中心点
            # 这比直接取均值更准确，因为越靠近中心预测通常越准
            weights = cluster_scores / np.sum(cluster_scores)
            weighted_center = np.sum(cluster_votes * weights[:, np.newaxis], axis=0)

            # 该簇的置信度取簇内最大分值（或者平均分）
            cluster_conf = np.max(cluster_scores)

            final_centers.append(weighted_center)
            final_scores.append(cluster_conf)

        batch_results.append(np.concatenate([np.array(final_centers), np.array(final_scores).reshape(-1, 1)], axis=-1))


    return batch_results


def decode_multi_landmarks(coords, heatmaps, offsets, k=20, threshold=0.1):
    """
    从单通道热图中提取多个局部峰值关键点
    :param coords: (B, N, 3) 原始坐标
    :param heatmaps: (B, N, 1) 热图分数
    :param offsets: (B, N, 3) 偏移量
    :param k: KNN 邻域大小
    :param threshold: 分数阈值
    :return: List of Tensors, 每个 Tensor 为 (M, 3), M 是检测到的点数
    """
    B, N, _ = coords.shape
    flat_heat = heatmaps.squeeze(-1)  # (B, N)
    flat_heat_numpy = np.sort(flat_heat.reshape(-1).detach().cpu().numpy())[::-1]

    # 1. 计算 KNN (处理大规模点云建议用更快的实现，如 pytorch3d)
    dist_matrix = torch.cdist(coords, coords)
    _, nn_idx = torch.topk(dist_matrix, k=k, largest=False)  # (B, N, k)

    # 2. 获取邻居分数并寻找局部最大值 (NMS)
    # neighbor_scores 形状: (B, N, k)
    neighbor_scores = torch.gather(flat_heat.unsqueeze(1).expand(-1, N, -1), 2, nn_idx)
    max_scores, _ = torch.max(neighbor_scores, dim=2)

    # 局部最大值掩码：当前点分值 >= 邻域最大值 且 超过阈值
    is_max = (flat_heat >= max_scores) & (flat_heat > threshold)  # (B, N)

    final_landmarks_list = []

    for b in range(B):
        # 3. 提取该 batch 中所有满足条件的索引
        indices = torch.nonzero(is_max[b]).squeeze(-1) # (M,)

        if indices.numel() == 0:
            final_landmarks_list.append(torch.empty((0, 3)).to(coords.device))
            continue

        # 4. 获取基础坐标 + 对应的偏移量
        base_pos = coords[b, indices]  # (M, 3)
        offset_vec = offsets[b, indices]  # (M, 3)
        socre = flat_heat[b, indices].unsqueeze(-1)
        # 最终位置修正公式：
        # $$P_{final} = P_{base} + \Delta P$$
        final_pos = base_pos + offset_vec

        final_landmarks_list.append(torch.cat([final_pos, socre], dim=-1).detach().cpu().numpy())



    return final_landmarks_list


def decoder_land(preheat_map, preoff_map, batch_data, pred_all, nums, conf_thresh):
    B, N, C = batch_data.shape
    batch_data = batch_data.repeat(cfg.landmarks_class, 1, 1)
    preheat_map = preheat_map.permute(0, 2, 1).reshape(cfg.landmarks_class*B, N, 1)
    preoff_map = preoff_map.permute(0, 2, 1, 3).reshape(cfg.landmarks_class*B, N, 3)

    #'Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'Cusp'
    #prelandmarks = decode_multi_landmarks(batch_data, preheat_map, preoff_map, k=20, threshold=conf_thresh)
    #prelandmarks = np.concatenate(prelandmarks, axis=0)
    preMesial = decode_with_nms_refined(batch_data[0:1, ...], preheat_map[0:1, ...], preoff_map[0:1, ...], score_thresh=conf_thresh, nms_radius=1.3, K=100)
    preDistal = decode_with_nms_refined(batch_data[1:2, ...], preheat_map[1:2, ...], preoff_map[1:2, ...], score_thresh=conf_thresh, nms_radius=1.3, K=100)
    preIPoint = decode_with_nms_refined(batch_data[2:3, ...], preheat_map[2:3, ...], preoff_map[2:3, ...], score_thresh=conf_thresh, nms_radius=2., K=100)
    preOPoint = decode_with_nms_refined(batch_data[3:4, ...], preheat_map[3:4, ...], preoff_map[3:4, ...], score_thresh=conf_thresh, nms_radius=1.3, K=100)
    preFPoint = decode_with_nms_refined(batch_data[4:5, ...], preheat_map[4:5, ...], preoff_map[4:5, ...], score_thresh=conf_thresh, nms_radius=1.3, K=100)
    preCusp = decode_with_nms_refined(batch_data[5:6, ...], preheat_map[5:6, ...], preoff_map[5:6, ...], score_thresh=conf_thresh, nms_radius=1.3, K=100)
    prelandmarks = preMesial+ preDistal+ preIPoint+ preOPoint+ preFPoint+preCusp
    # prelandmarks, presocres = prelandmarks.detach().cpu().numpy(), presocres.detach().cpu().numpy()
    # file_ = open("./outputs/landmarks_select.txt", "w")
    # for i in range(prelandmarks.shape[0]):
    #     point = prelandmarks[i]
    #     file_.write(str(point[0])+" " + str(point[1])+" " +str(point[2])+" \n")
    # file_.close()


    #prelandmarks = decode_teeth_with_dbscan(batch_data, preheat_map, preoff_map,score_thresh=0.1,eps=0.8, min_samples=9)
    #
    # mask = preheat_map > conf_thresh
    # predland = (batch_data + preoff_map)[mask]


    #######################################################################
    # pred_land = np.concatenate([fpt_land, mpt_land], axis=0)

    score = [prelandmarks[pi][:, -1] for pi in range(len(cfg.landPorder))]
    kp = [prelandmarks[pi][:, :3] for pi in range(len(cfg.landPorder))]
    for pi in range(len(cfg.landPorder)):
        pred_all[cfg.landPorder[pi]][nums] = [tuple(x) for x in zip(kp[pi], score[pi])]


    return pred_all


def assign_colors_by_probability(points, probabilities):
    """根据概率值赋予蓝色渐变（浅蓝→深蓝）"""
    probabilities = np.clip(probabilities, 0, 1).flatten()
    colors = np.zeros((len(probabilities), 3))
    colors[:, 2] = np.round(255 * probabilities)  # 蓝色通道随概率增加
    colors[:, 0] = np.round(255 * (1 - probabilities))  # 红色通道递减（可选）
    colors[:, 1] = np.round(255 * (1 - probabilities))  # 绿色通道递减（可选）
    return np.hstack((points, colors.astype(int)))


def save_colored_points(filename, colored_points):
    """
    保存带颜色的点云到txt文件

    参数:
        filename: 输出文件名
        colored_points: Nx6数组，前3列是坐标，后3列是RGB值
    """
    # 确保RGB是整数
    colored_points[:, 3:] = np.round(colored_points[:, 3:])

    # 保存为txt文件，使用空格分隔
    np.savetxt(filename, colored_points, fmt='%.6f %.6f %.6f %d %d %d')

def cal_acc_seg_land(model, test_loader, conf_thresh=0.1):

    gt_all =  { key:{} for key in (cfg.landPorder)}
    pred_all = { key:{} for key in (cfg.landPorder)}
    pred_alld = { key:{} for key in (cfg.landPorder)}
    # pred_all: map of {classname: {meshname: [(kp, score)]}}
    # gt_all: map of {classname: {meshname: [kp]}}

    fm_cd, fm_cdd = [], []
    nums = 0
    for batch_data, label_landmarks in test_loader:
        # label_landmarks = add_gaussian_noise(label_landmarks)

        nums = nums + 1
        batch_data = batch_data.cuda().float()


        with torch.no_grad():
            # batch_data = add_gaussion_noise(batch_data)
            preheat_map, preoff_map = model(batch_data)
            ######################################################
            # preheat_map_numpy = preheat_map.squeeze().detach().cpu().numpy()
            # batch_data_numpy = batch_data.squeeze().detach().cpu().numpy()
            # for i in range(preheat_map_numpy.shape[-1]):
            #     colored_points = assign_colors_by_probability(batch_data_numpy, preheat_map_numpy[..., i])
            #     save_colored_points("./outputs/heatmap" +str(i)+".txt", colored_points)

            pred_all = decoder_land(preheat_map, preoff_map, batch_data, pred_all, nums, conf_thresh)

            # gt_land = []
            # for i , key_class in enumerate(label_landmarks):
            #     key_point = label_landmarks[key_class].reshape(-1, 3).numpy()
            #     label_landmarks[key_class] = key_point
            #     gt_land.append(key_point)
            # gt_land = np.concatenate(gt_land, axis=0)


            # cdv = chamfer_distance(gt_land, pred_land)
            # fm_cd.append(cdv)
            # cdvd = chamfer_distance(gt_land, pred_landd)
            # fm_cdd.append(cdvd)

            for pi in range(len(cfg.landPorder)):
                if label_landmarks[cfg.landPorder[pi]] ==[]:
                    continue
                gt_all[cfg.landPorder[pi]][nums] = label_landmarks[cfg.landPorder[pi]].squeeze().cpu().numpy()

            #######################################################################

    all_metrics = cal_score(gt_all, pred_all)
    #all_metricsd = cal_score(gt_all, pred_alld)
    #################################
    fm_cdv = np.mean(np.array(fm_cd))
    fm_cdvd = np.mean(np.array(fm_cdd))

    scores = reformat_scores(all_metrics)
    #scoresd = reformat_scores(all_metricsd)
    fm_cdvd, all_metricsd, scoresd = fm_cdv, all_metrics, scores
    return fm_cdv, all_metrics, scores, fm_cdvd, all_metricsd, scoresd


def decode_with_nms_refined_cls(points, heatmap, distance_map, cls, score_thresh=0.5, nms_radius=1.5, K=100):
    """
    使用 3D 距离 NMS + 权重质心提炼获取牙齿关键点

    Args:
        points: (B, N, 3) 原始点云输入
        heatmap: (B, N, 1) 每个点的热图分数
        distance_map: (B, N, 3) 每个点到其对应中心点的偏移向量
        score_thresh: 分数过滤阈值
        nms_radius: 抑制半径 (单位通常为 mm)
        K: 每个样本最多保留的关键点数量
    """
    B, N, _ = points.shape
    device = points.device

    # 1. 计算预测的目标绝对中心位置: P_center = P_origin + Offset
    pred_centers = points + distance_map  # (B, N, 3)
    scores = heatmap.squeeze(-1)  # (B, N)
    cls = cls.squeeze(-1)  # (B, N)

    batch_final_results = []
    nms_radius_sq = nms_radius ** 2  # 预计算平方，避免循环内开方
    IOPoints = {0:[], 1:[]}
    for b in range(B):
        cur_scores = scores[b]
        cur_centers = pred_centers[b]
        cur_cls = cls[b]
        # 2. 阈值过滤
        mask = cur_scores > score_thresh
        if not mask.any():
            batch_final_results.append(np.zeros((0, 4)))  # [x, y, z, score]
            continue

        valid_scores = cur_scores[mask]
        valid_centers = cur_centers[mask]
        valid_cls = cur_cls[mask]
        # 3. 按照分数从大到小排序
        indices = torch.argsort(valid_scores, descending=True)
        valid_scores = valid_scores[indices]
        valid_centers = valid_centers[indices]
        valid_cls = valid_cls[indices]

        # 4. 带有中心提炼的贪心 NMS
        refined_centers = []
        refined_scores = []
        is_suppressed = torch.zeros(len(valid_scores), dtype=torch.bool, device=device)

        for i in range(len(valid_scores)):
            if is_suppressed[i]:
                continue

            # 计算当前最高分点与所有（未抑制）点的距离
            # 注意：这里我们计算与全集的距离，以便找出所有邻居进行加权
            dist_sq = torch.sum((valid_centers[i:i + 1] - valid_centers) ** 2, dim=-1)

            # 找出半径内的所有邻居点
            # 只有尚未被前面簇吸收的点才参与当前簇的计算
            neighbor_mask = (dist_sq < nms_radius_sq) & (~is_suppressed)
            neighbor_mask_ = (dist_sq < nms_radius_sq*0.3) & (~is_suppressed)
            # --- 核心改进：加权质心提炼 ---
            # 提取邻域内的所有点和分数
            cluster_centers = valid_centers[neighbor_mask_]
            cluster_scores = valid_scores[neighbor_mask_].unsqueeze(-1)  # (M, 1)

            # 计算加权平均位置：Sum(Pos * Score) / Sum(Score)
            # 这样预测越准的点（分数越高），对最终位置的影响力越大
            weights = cluster_scores / (cluster_scores.sum() + 1e-6)
            refined_pos = (cluster_centers * weights).sum(dim=0)

            clids = torch.argmax(valid_cls[i]).item()
            # 记录结果：位置用提炼后的，分数通常保留该簇的最大值
            refined_centers.append(refined_pos)
            refined_scores.append(valid_scores[i])
            IOPoints[clids].append(torch.cat([refined_pos, valid_scores[i].reshape(1)], dim=0).cpu().numpy())

            # 抑制该半径内所有的点，防止被重复计算
            is_suppressed[neighbor_mask] = True

            if len(refined_centers) >= K:
                break

        if len(refined_centers) > 0:
            res_centers = torch.stack(refined_centers)
            res_scores = torch.stack(refined_scores).unsqueeze(-1)
            # 合并为 (M, 4) 矩阵 [x, y, z, score]
            batch_result = torch.cat([res_centers, res_scores], dim=-1).detach().cpu().numpy()
        else:
            batch_result = np.zeros((0, 4))

        batch_final_results.append(batch_result)
    inpoint = [np.array(IOPoints[0])]
    onpoint = [np.array(IOPoints[1])]
    return inpoint, onpoint


def decoder_land_cls(preheat_map, preoff_map, cls, batch_data, pred_all, nums, conf_thresh):
    B, N, C = batch_data.shape

    preheat_map = preheat_map.squeeze().permute(1, 0)
    preoff_map = preoff_map.permute(0, 2, 1, 3).reshape((cfg.landmarks_class-1)*B, N, 3)

    cls = cls[..., 2:4]
    preMesial = decode_with_nms_refined(batch_data, preheat_map[0:1, ...], preoff_map[0:1, ...], score_thresh=conf_thresh, nms_radius=0.7, K=100)
    preDistal = decode_with_nms_refined(batch_data, preheat_map[1:2, ...], preoff_map[1:2, ...], score_thresh=conf_thresh, nms_radius=0.7, K=100)
    preIPoint, preOPoint = decode_with_nms_refined_cls(batch_data, preheat_map[2:3, ...], preoff_map[2:3, ...], cls, score_thresh=conf_thresh, nms_radius=1.0, K=200)

    preFPoint = decode_with_nms_refined(batch_data, preheat_map[3:4, ...], preoff_map[3:4, ...], score_thresh=conf_thresh, nms_radius=0.7, K=100)
    preCusp = decode_with_nms_refined(batch_data, preheat_map[4:5, ...], preoff_map[4:5, ...], score_thresh=conf_thresh, nms_radius=0.7, K=100)


    # preMesial = decode_with_nms_refined(batch_data, preheat_map[0:1, ...], preoff_map[0:1, ...], score_thresh=conf_thresh, nms_radius=0.7, K=100)
    # preDistal = decode_with_nms_refined(batch_data, preheat_map[1:2, ...], preoff_map[1:2, ...], score_thresh=conf_thresh, nms_radius=0.7, K=100)
    # preIPoint = decode_with_nms_refined(batch_data, preheat_map[2:3, ...], preoff_map[2:3, ...], score_thresh=conf_thresh, nms_radius=0.7, K=100)
    # preOPoint = decode_with_nms_refined(batch_data, preheat_map[3:4, ...], preoff_map[3:4, ...], score_thresh=conf_thresh, nms_radius=0.7, K=100)
    # preFPoint = decode_with_nms_refined(batch_data, preheat_map[4:5, ...], preoff_map[4:5, ...], score_thresh=conf_thresh, nms_radius=0.7, K=100)
    # preCusp = decode_with_nms_refined(batch_data, preheat_map[5:6, ...], preoff_map[5:6, ...], score_thresh=conf_thresh, nms_radius=0.7, K=100)



    # prelandmarks = preMesial+ preDistal+ preIPoint+ preOPoint+ preFPoint+preCusp
    # preheat_map_numpy = preheat_map.squeeze().detach().cpu().numpy()
    # batch_data_numpy = batch_data.squeeze().detach().cpu().numpy()
    # for i in range(preheat_map_numpy.shape[0]):
    #     colored_points = assign_colors_by_probability(batch_data_numpy, preheat_map_numpy[i])
    #     save_colored_points("./outputs/heatmap" +str(i)+".txt", colored_points)
    #
    #
    # for i in range(len(prelandmarks)):
    #     prd_landmarks_numpy = prelandmarks[i]
    #     colored_points = assign_colors_by_probability(prd_landmarks_numpy[:, :3], prd_landmarks_numpy[:, 3])
    #     save_colored_points("./outputs/landmarks" +str(i)+".txt", colored_points)

    score = [prelandmarks[pi][:, -1] for pi in range(len(cfg.landPorder))]
    kp = [prelandmarks[pi][:, :3] for pi in range(len(cfg.landPorder))]
    for pi in range(len(cfg.landPorder)):
        pred_all[cfg.landPorder[pi]][nums] = [tuple(x) for x in zip(kp[pi], score[pi])]


    return pred_all



def cal_acc_seg_land_cls(model, test_loader, conf_thresh=0.1):

    gt_all =  { key:{} for key in (cfg.landPorder)}
    pred_all = { key:{} for key in (cfg.landPorder)}

    fm_cd, fm_cdd = [], []
    nums = 0
    for batch_data, label_landmarks in test_loader:

        nums = nums + 1
        batch_data = batch_data.cuda().float()

        with torch.no_grad():
            # batch_data = add_gaussion_noise(batch_data)
            preheat_map, preoff_map, cls = model(batch_data)
            ######################################################
            # preheat_map_numpy = preheat_map.squeeze().detach().cpu().numpy()
            # batch_data_numpy = batch_data.squeeze().detach().cpu().numpy()
            # for i in range(preheat_map_numpy.shape[-1]):
            #     colored_points = assign_colors_by_probability(batch_data_numpy, preheat_map_numpy[..., i])
            #     save_colored_points("./outputs/heatmap" +str(i)+".txt", colored_points)

            pred_all = decoder_land_cls(preheat_map, preoff_map, cls, batch_data, pred_all, nums, conf_thresh)

            for pi in range(len(cfg.landPorder)):
                if label_landmarks[cfg.landPorder[pi]] ==[]:
                    continue
                gt_all[cfg.landPorder[pi]][nums] = label_landmarks[cfg.landPorder[pi]].squeeze().cpu().numpy()

            #######################################################################

    all_metrics = cal_score(gt_all, pred_all)

    #################################
    fm_cdv = np.mean(np.array(fm_cd))
    scores = reformat_scores(all_metrics)

    return fm_cdv, all_metrics, scores

from scipy.spatial import cKDTree
def merge_points_by_distance(kp, score, th=0.7*0.7*0.3):
    """
    按距离和得分加权合并关键点，合并后的得分为局部均值。

    参数:
    kp: np.ndarray, shape (N, 3), 关键点坐标
    score: np.ndarray, shape (N,), 关键点得分
    th: float, 距离阈值

    返回:
    merged_kp: np.ndarray, shape (M, 3), 合并后的关键点坐标
    merged_score: np.ndarray, shape (M,), 合并后的得分 (取均值)
    """
    N = len(kp)
    if N == 0:
        return np.empty((0, 3)), np.empty((0,))

    # 1. 按照得分从高到低排序的索引
    order = np.argsort(score)[::-1]

    # 记录哪些点已经被合并
    visited = np.zeros(N, dtype=bool)

    merged_kps = []
    merged_scores = []

    # 2. 建立 KD-Tree 加速空间查询
    tree = cKDTree(kp)

    for idx in order:
        if visited[idx]:
            continue

        # 3. 查找距离当前最高分点小于等于阈值 th 的所有点的索引
        close_indices = tree.query_ball_point(kp[idx], th)

        # 过滤掉已经被合并过的点
        valid_close_indices = [i for i in close_indices if not visited[i]]

        if not valid_close_indices:
            continue

        # 提取这些点的坐标和得分
        valid_close_indices = np.array(valid_close_indices)
        p_close = kp[valid_close_indices]
        s_close = score[valid_close_indices]

        # 4. 根据得分计算坐标的加权平均
        weight_sum = np.sum(s_close)
        merged_p = np.sum(p_close * s_close[:, np.newaxis], axis=0) / (weight_sum + 1e-8)

        # 5. 【更新】计算合并后的新得分：取所有参与合并点的均值
        merged_s = np.mean(s_close)

        merged_kps.append(merged_p)
        merged_scores.append(merged_s)

        # 6. 将这些点标记为已合并
        visited[valid_close_indices] = True

    return np.array(merged_kps), np.array(merged_scores)

def get_valid_landmarks(pred_landmarks, reg_conf, pred_all, nums, conf_threshold=0.1):
    typenum = {'Mesial': 50, 'Distal': 50, 'InnerPoint': 50, 'OuterPoint': 50, 'FacialPoint': 50, 'Cusp': 50}
    # 确保输入是numpy数组
    prelandmarks = pred_landmarks.squeeze(dim=0).detach().cpu().numpy()
    reg_conf = reg_conf.squeeze(dim=0).detach().cpu().numpy()

    # 初始化结果字典
    score = [reg_conf[pi][:, -1] for pi in range(len(cfg.landPorder))]
    kp = [prelandmarks[pi][:, :3] for pi in range(len(cfg.landPorder))]
    for pi in range(len(cfg.landPorder)):
        kn = typenum[cfg.landPorder[pi]]
        #merged_kp, merged_score = merge_points_by_distance(kp[pi], score[pi], th=0.05)

        pred_all[cfg.landPorder[pi]][nums] = [tuple(x) for x in zip(kp[pi][:kn, ...], score[pi][:kn, ...])]

    return pred_all

from scipy.optimize import linear_sum_assignment
def cal_acc_seg_land_reg(model, test_loader, conf_thresh=0.1):

    gt_all =  { key:{} for key in (cfg.landPorder)}
    pred_all = { key:{} for key in (cfg.landPorder)}
    cd_all =  { key:[] for key in (cfg.landPorder)}
    fm_cd, fm_cdd = [], []
    nums = 0
    for batch_data, label_landmarks, file_path in test_loader:

        nums = nums + 1
        batch_data = batch_data.cuda().float()

        with torch.no_grad():
            # batch_data = add_gaussion_noise(batch_data)
            out_seg, preheat_map, preoff_map, cls, prd_landmarks, pre_conf, pred_delta, select_pos_reshaped = model(batch_data)
            ######################################################
            #pre_conf = pre_conf.sigmoid()

            # preheat_map_numpy = preheat_map.squeeze().detach().cpu().numpy()
            # batch_data_numpy = batch_data.squeeze().detach().cpu().numpy()
            # for i in range(preheat_map_numpy.shape[-1]):
            #     colored_points = assign_colors_by_probability(batch_data_numpy, preheat_map_numpy[..., i])
            #     save_colored_points("./outputs/heatmap" +str(i)+".txt", colored_points)
            #
            # prd_landmarks_numpy = select_pos_reshaped.squeeze().detach().cpu().numpy()
            # pre_conf_numpy = pre_conf.squeeze().detach().cpu().numpy()
            # for i in range(prd_landmarks_numpy.shape[0]):
            #     colored_points = assign_colors_by_probability(prd_landmarks_numpy[i], pre_conf_numpy[i])
            #     save_colored_points("./outputs/landmarks" +str(i)+".txt", colored_points)

            pred_all = get_valid_landmarks(select_pos_reshaped, pre_conf, pred_all, nums, conf_threshold=0.1)

            # pred_numpy = torch.cat([prd_landmarks, pre_conf], dim=-1).squeeze().detach().cpu().numpy()
            # np.save(file_path[0].replace("_c.npy", "") + "_ours.npy", pred_numpy)
            for pi in range(len(cfg.landPorder)):
                if label_landmarks[cfg.landPorder[pi]] ==[]:
                    continue
                gt_all[cfg.landPorder[pi]][nums] = label_landmarks[cfg.landPorder[pi]].squeeze().cpu().numpy()


            for c_idx, c_name in enumerate(pred_all):
                pt = pred_all[c_name][nums]
                gt = gt_all[c_name][nums]

                pts = np.array([pt[i][0] for i in range(len(pt))])
                score = np.array([pt[i][1] for i in range(len(pt))])
                if len(gt) >=1:
                    gt = gt.reshape(-1,3)
                    cost_matrix = torch.cdist(torch.tensor(pts).float(), torch.tensor(gt).float(), p=2)  # (K, M)
                    pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
                    pts = pts[pred_indices]

                    cd = chamfer_distance(pts, gt)
                    cd_all[c_name].append(cd)
                    score = score[pred_indices]

                #     pred_all[c_name][nums] = [(pts[i], score[i]) for i in range(pts.shape[0])]
                # else:
                #     pred_all[c_name][nums] = [(pt[i][0], 0) for i in range(len(pt))]

        #     for c_idx, c_name in enumerate(pred_all):
        #         pt = pred_all[c_name][nums]
        #         pts = np.array([pt[i][0] for i in range(len(pt))])
        #         score = np.array([pt[i][1] for i in range(len(pt))])
        #         colored_points = assign_colors_by_probability(pts, score)
        #         save_colored_points("./outputs/landmarks" +str(c_idx)+".txt", colored_points)
        # print("voer")
            #######################################################################

    all_metrics = cal_score(gt_all, pred_all)

    #################################
    fm_cdv = np.mean(np.array(fm_cd))
    scores = reformat_scores(all_metrics)

    return fm_cdv, all_metrics, scores, cd_all







from test_oneheat_reg import get_weighted_centers
def cal_acc_seg_land_oneheat_reg(model, test_loader, conf_thresh=0.1):

    gt_all =  { key:{} for key in (cfg.landPorder)}
    pred_all = { key:{} for key in (cfg.landPorder)}
    cd_all =  { key:[] for key in (cfg.landPorder)}
    fm_cd, fm_cdd = [], []
    nums = 0
    for batch_data, label_landmarks, file_path in test_loader:

        nums = nums + 1
        batch_data = batch_data.cuda().float()

        with torch.no_grad():
            # batch_data = add_gaussion_noise(batch_data)
            preheat_map, preoff_map, cls, init_land, init_conf, shifted_pos_all,twoheatmap, twooffset_raw, twoseg_mask, final_centers, relative_pos = model(batch_data)

            local_pos = final_centers.squeeze(dim=0).unsqueeze(dim=1) + relative_pos
            pre_points = get_weighted_centers(twoheatmap, twooffset_raw, local_pos, k=20, topk=1, dist_thresh=1.0)

            fixed = pre_points["fixed"].transpose(1, 0, 2, 3).reshape(6, -1, 4)
            cusps = pre_points["cusps"].reshape(-1, 4)
            cusps = cusps[cusps[:, -1] >0.1]


            score = [fixed[pi, :, 3].reshape(-1) for pi in range(len(cfg.landPorder)-1)]
            kp = [fixed[pi, :, :3].reshape(-1, 3)  for pi in range(len(cfg.landPorder)-1)]
            score.append(cusps[:, 3].reshape(-1))
            kp.append(cusps[:, :3].reshape(-1, 3))

            for pi in range(len(cfg.landPorder)):

                pred_all[cfg.landPorder[pi]][nums] = [tuple(x) for x in zip(kp[pi], score[pi])]

            # pred_numpy = torch.cat([prd_landmarks, pre_conf], dim=-1).squeeze().detach().cpu().numpy()
            # np.save(file_path[0].replace("_c.npy", "") + "_ours.npy", pred_numpy)
            for pi in range(len(cfg.landPorder)):
                if label_landmarks[cfg.landPorder[pi]] ==[]:
                    continue
                gt_all[cfg.landPorder[pi]][nums] = label_landmarks[cfg.landPorder[pi]].squeeze().cpu().numpy()

            for c_idx, c_name in enumerate(pred_all):
                pt = pred_all[c_name][nums]
                gt = gt_all[c_name][nums]

                pts = np.array([pt[i][0] for i in range(len(pt))])
                score = np.array([pt[i][1] for i in range(len(pt))])
                if len(gt) >=1:
                    gt = gt.reshape(-1,3)
                    cost_matrix = torch.cdist(torch.tensor(pts).float(), torch.tensor(gt).float(), p=2)  # (K, M)
                    pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
                    pts = pts[pred_indices]

                    cd = chamfer_distance(pts, gt)
                    cd_all[c_name].append(cd)
                    score = score[pred_indices]




        #     for c_idx, c_name in enumerate(pred_all):
        #         pt = pred_all[c_name][nums]
        #         pts = np.array([pt[i][0] for i in range(len(pt))])
        #         score = np.array([pt[i][1] for i in range(len(pt))])
        #         colored_points = assign_colors_by_probability(pts, score)
        #         save_colored_points("./outputs/landmarks" +str(c_idx)+".txt", colored_points)
        # print("voer")
            #######################################################################

    all_metrics = cal_score(gt_all, pred_all)

    #################################
    fm_cdv = np.mean(np.array(fm_cd))
    scores = reformat_scores(all_metrics)

    return fm_cdv, all_metrics, scores, cd_all






def chamfer_distance(points1, points2):
    # 计算每个点到另一组点云的最近邻距离
    dist1 = np.sqrt(np.sum((points1[:, None, :] - points2[None, :, :]) ** 2, axis=-1))
    dist2 = np.sqrt(np.sum((points2[:, None, :] - points1[None, :, :]) ** 2, axis=-1))

    # 计算每个点到另一组点云的最近邻距离之和
    cd = np.sum(np.min(dist1, axis=1)) + np.sum(np.min(dist2, axis=1))

    # 计算平均 Chamfer Distance
    avg_cd = cd / (points1.shape[0] + points2.shape[0])

    return avg_cd

