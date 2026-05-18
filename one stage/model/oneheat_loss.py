
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from config import oneheat_config as cfg


class WingLoss(nn.Module):
    def __init__(self, omega=0.01, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)

    def forward(self, pred, target, weights=None):
        delta_2 = (target - pred).pow(2).sum(dim=-1, keepdim=False)
        delta = delta_2.clamp(min=1e-6).sqrt()

        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        if weights is not None:
            weighted_loss = loss * weights
            return weighted_loss.sum() / (weights.sum() + 1e-6)
        else:
            return loss.mean()


def get_tooth_cenp(target_dict):

    tooth_kep = {}
    tooth_cenp = torch.zeros((len(target_dict), 3))
    for i, tid in enumerate(target_dict):
        kep, _ = target_dict[tid]
        cenp = kep["cenpseg"]  if "cenpseg" in kep else kep["cenpland"]
        tooth_cenp[i] = torch.tensor(cenp)

        mask = torch.zeros((cfg.kpnums)).long()
        fixed_kep, match_kep = torch.zeros((6, 3)), None
        for ki, kcls in enumerate(cfg.landPorder):
            if (kcls in kep) and ("Cusp" != kcls):
                fixed_kep[ki] = kep[kcls].reshape(3)
                mask[ki] =1
            elif(kcls in kep) and ("Cusp" == kcls):
                match_kep = kep[kcls].reshape(-1, 3)
                #mask[ki:ki+match_kep.shape[0]]=1

        mask[5] = 1
        fixed_kep[-1] = cenp
        tooth_kep[i] = {"fixed": fixed_kep, "match": match_kep, "mask":mask}

    return tooth_cenp.cuda().float(), tooth_kep

def cal_tooth_confidence_losss(gtooth_cenp, shifted_pos_all, pconf, dtype, sigma=2.0):
    with torch.no_grad():
        # 假设 shifted_pos_all 维度是 [1, T, 3] -> squeeze后是 [T, 3]
        dist_matrix = torch.cdist(shifted_pos_all.squeeze(dim=0), gtooth_cenp, p=2)

        gt_to_sel_dist, gt_to_sel_idx = dist_matrix.min(dim=0)
        sel_to_gt_dist, sel_to_gt_idx = dist_matrix.min(dim=1)

        # 计算软标签分数 (反映位置精度) -> [T]
        soft_scores = torch.exp(-(sel_to_gt_dist ** 2) / (2 * sigma ** 2))

    # 【修正】: 确保 pconf 挤压到 1D，防止 MSE 广播错误 [T, 1] vs [T]
    loss = F.mse_loss(pconf.view(-1).to(dtype), soft_scores.to(dtype))


    return loss, soft_scores, sel_to_gt_idx,sel_to_gt_dist


class OneMultiInstanceMatchLoss(nn.Module):
    def __init__(self, class_names=None, sigma=1.5, dist_th=3.0):
        super(OneMultiInstanceMatchLoss, self).__init__()
        self.th = 0.1
        self.sigma = sigma
        self.dist_th = dist_th  # 距离阈值 3mm
        self.wingloss = WingLoss()

    def quality_focal_loss(self, pred_logits, target_scores, beta=2.0):
        pred_sigmoid = torch.sigmoid(pred_logits)
        pred_sigmoid = torch.clamp(pred_sigmoid, 1e-6, 1.0 - 1e-6)

        loss = - (target_scores * torch.log(pred_sigmoid) +
                  (1 - target_scores) * torch.log(1 - pred_sigmoid))

        weight = torch.pow(torch.abs(target_scores - pred_sigmoid), beta)
        return (loss * weight).mean()

    def forward(self, pconf, pland, shifted_pos_all, target_dict):
        device = pconf.device
        dtype = pconf.dtype

        # 获取真值中心点
        gtooth_cenp, tooth_kep = get_tooth_cenp(target_dict)
        gtooth_cenp = gtooth_cenp.to(device)

        # 1. 计算牙齿中心点置信度损失，并获取匹配索引和距离
        tooth_conf_loss, soft_scores, sel_to_gt_idx, sel_to_gt_dist = cal_tooth_confidence_losss(
            gtooth_cenp, shifted_pos_all, pconf, dtype, self.sigma)

        # 2. 计算 pland 的回归损失 (仅针对距离真值 < 3mm 的预测点)
        # 确保 pland 维度为 [T, 3]
        pred_pland = (shifted_pos_all +pland).view(-1, 3)

        # 根据匹配索引获取对应的真值中心点
        target_cenp = gtooth_cenp[sel_to_gt_idx]  # [T, 3]

        # 创建有效掩码：距离小于阈值的点参与回归
        valid_mask = sel_to_gt_dist < self.dist_th  # [T]

        if valid_mask.any():
            # 提取有效预测和对应的真值
            valid_pred = pred_pland[valid_mask]
            valid_target = target_cenp[valid_mask]

            # 计算距离匹配损失 (WingLoss)
            dist_reg_loss = self.wingloss(valid_pred, valid_target)
        else:
            # 如果没有匹配上的点，损失设为 0
            dist_reg_loss = torch.tensor(0.0, device=device, dtype=dtype)

        # 总损失权重分配 (示例权重，可自行调整)
        total_loss = tooth_conf_loss + 1.0 * dist_reg_loss

        return total_loss, tooth_conf_loss, dist_reg_loss


# class OneMultiInstanceMatchLoss(nn.Module):
#     def __init__(self, sigma=2.0, dist_th=3.0, reg_weight=2.0):
#         super(OneMultiInstanceMatchLoss, self).__init__()
#         self.sigma = sigma
#         self.dist_th = dist_th  # 仅用于限制坐标回归的范围
#         self.reg_weight = reg_weight
#         self.wingloss = WingLoss()
#
#     def quality_focal_loss(self, pred_logits, target_scores, beta=2.0):
#         pred_sigmoid = torch.sigmoid(pred_logits)
#         pred_sigmoid = torch.clamp(pred_sigmoid, 1e-5, 1.0 - 1e-5)
#         loss = - (target_scores * torch.log(pred_sigmoid) +
#                   (1 - target_scores) * torch.log(1 - pred_sigmoid))
#         weight = torch.pow(torch.abs(target_scores - pred_sigmoid), beta)
#         return (loss * weight).mean()
#
#     def forward(self, pconf, pland, shifted_pos_all, target_dict):
#         device = pconf.device
#         dtype = pconf.dtype
#
#         # 1. 获取 GT 中心点
#         gtooth_cenp, _ = get_tooth_cenp(target_dict)
#         gtooth_cenp = gtooth_cenp.to(device)  # [N_gt, 3]
#
#         anchor_pos = shifted_pos_all.view(-1, 3)
#         pconf_flat = pconf.view(-1)
#         pland_flat = pland.view(-1, 3)
#
#         if gtooth_cenp.shape[0] == 0:
#             return self.quality_focal_loss(pconf_flat, torch.zeros_like(pconf_flat)), torch.zeros(1), torch.zeros(1)
#
#         # 2. 匹配与距离计算
#         with torch.no_grad():
#             dist_matrix = torch.cdist(anchor_pos, gtooth_cenp, p=2)
#             min_dist, nearest_gt_idx = dist_matrix.min(dim=1)  # [N_anchor]
#
#         # ---------------------------------------------------------
#         # 3. 吸收你的精髓：全局置信度软标签 (无硬性截断，防止漏检)
#         # ---------------------------------------------------------
#         # 即使距离 > 3mm，依然给一个微小的高斯概率，维持正向梯度，防止被完全抹杀
#         target_conf = torch.exp(-(min_dist ** 2) / (2 * self.sigma ** 2)).to(dtype)
#
#         # 计算置信度损失 (如果你发现QFL还是压制太狠，这里可以直接换回你的 F.mse_loss)
#         tooth_conf_loss = self.quality_focal_loss(pconf_flat, target_conf)
#
#         # ---------------------------------------------------------
#         # 4. 回归的保护伞：局部阈值掩码 (仅距离近的点参与坐标微调)
#         # ---------------------------------------------------------
#         pos_mask = min_dist < self.dist_th  # 3.0mm 内的点负责回归
#
#         if pos_mask.any():
#             pred_final_pos = anchor_pos[pos_mask] + pland_flat[pos_mask]
#             target_pos = gtooth_cenp[nearest_gt_idx[pos_mask]]
#             dist_reg_loss = self.wingloss(pred_final_pos, target_pos)
#         else:
#             dist_reg_loss = torch.tensor(0.0, device=device, dtype=dtype)
#
#         # 5. 总损失
#         total_loss = tooth_conf_loss + self.reg_weight * dist_reg_loss
#
#         return total_loss, tooth_conf_loss, dist_reg_loss

class ToothHeatmapLoss(nn.Module):
    """
    适用于点云热图的连续型 Focal Loss。
    正样本 Mask 由热图真值阈值决定。
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super(ToothHeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt, mask):
        """
        pred: (K, 6)
        gt:   (K, 6)
        mask: (K, 6) - gt > 0.01 的点
        """
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        # 连续型 Focal Loss 公式
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred) * torch.pow(gt, self.beta)
        neg_loss = -torch.pow(pred, self.alpha) * torch.log(1 - pred) * torch.pow(1 - gt, self.beta)

        # 使用传入的 mask 进行索引
        p_loss = pos_loss[mask].mean() if mask.any() else torch.tensor(0.0, device=pred.device)
        n_loss = neg_loss[~mask].mean() if (~mask).any() else torch.tensor(0.0, device=pred.device)

        return p_loss + n_loss


class Stage2LocalLandmarkLoss(nn.Module):
    def __init__(self, dist_th=3.0, sigma=0.7, k_offset=40, hm_th=0.01):
        super(Stage2LocalLandmarkLoss, self).__init__()
        self.dist_th = dist_th
        self.sigma = sigma
        self.k_offset = k_offset
        self.hm_th = hm_th

    def forward(self, final_centers, relative_pos, twoheatmap, twooffset_raw, target_dict_list):
        """
        final_centers: [B, max_teeth, 3] 第一阶段预测并 NMS 后的中心
        relative_pos: [B*max_teeth, 1536, 3] 局部采样点相对于 final_centers 的坐标
        twoheatmap: [B*max_teeth, 1536, 7]
        twooffset_raw: [B*max_teeth, 1536, 7, 3]
        """
        device = twoheatmap.device
        dtype = twoheatmap.dtype
        B = final_centers.shape[0]
        max_teeth = final_centers.shape[1]
        num_kps = twoheatmap.shape[-1]  # 应为 7

        stage2_hm_loss_total = 0.0
        stage2_off_loss_total = 0.0
        valid_crops_count = 0

        for b in range(B):
            target_dict = target_dict_list[b]
            # 获取当前 batch 所有 GT 牙齿的中心和关键点信息
            gtooth_cenp, tooth_kep = get_tooth_cenp(target_dict)
            gtooth_cenp = gtooth_cenp.to(device)

            batch_centers = final_centers[b]
            # 建立预测中心与 GT 中心的匹配 (Nearest Neighbor)
            dist_matrix = torch.cdist(batch_centers, gtooth_cenp, p=2)
            if dist_matrix.numel() == 0: continue

            c2g_dist, c2g_idx = dist_matrix.min(dim=1)

            for t_idx in range(max_teeth):
                idx_in_batch = b * max_teeth + t_idx
                pred_hm = twoheatmap[idx_in_batch]  # [1536, 7]
                pred_off = twooffset_raw[idx_in_batch]  # [1536, 7, 3]
                rel_pos = relative_pos[idx_in_batch]  # [1536, 3]
                pred_center_abs = batch_centers[t_idx]

                # --- 1. 背景裁剪 (远离任何 GT 牙齿) ---
                if c2g_dist[t_idx] > self.dist_th:
                    # 背景区域热图目标全为 0
                    stage2_hm_loss_total += F.mse_loss(pred_hm, torch.zeros_like(pred_hm))
                    continue

                # --- 2. 正样本裁剪 ---
                valid_crops_count += 1
                gt_id = c2g_idx[t_idx].item()
                gt_info = tooth_kep[gt_id]

                target_hm = torch.zeros_like(pred_hm)
                target_off = torch.zeros_like(pred_off)
                offset_mask = torch.zeros((rel_pos.shape[0], num_kps), device=device, dtype=torch.bool)

                # A. 固定点(0-4) + 中心点(5)
                # 这 6 个点在 GT 中是单点，统一处理
                fixed_indices = range(6)
                gt_fixed_abs = gt_info["fixed"].to(device).to(dtype)

                for k_id in fixed_indices:
                    # 检查该关键点是否存在 (mask[k_id] == 1)
                    if gt_info["mask"][k_id] == 1:
                        # 转换成相对于当前预测中心的局部坐标
                        kp_target_rel = gt_fixed_abs[k_id] - pred_center_abs
                        # 计算 1536 个采样点到该关键点目标的距离
                        dists = torch.norm(rel_pos - kp_target_rel, dim=-1)

                        # 生成高斯热图: $e^{-d^2 / 2\sigma^2}$
                        target_hm[:, k_id] = torch.exp(-(dists ** 2) / (2 * self.sigma ** 2))

                        # 偏移量目标: 目标关键点坐标 - 采样点坐标
                        target_off[:, k_id, :] = kp_target_rel - rel_pos

                        # Offset 掩码：只在关键点附近的 Top-K 个点计算回归损失
                        k_real = min(self.k_offset, dists.shape[0])
                        _, topk_idx = torch.topk(dists, k=k_real, largest=False)
                        offset_mask[topk_idx, k_id] = True

                # B. 牙尖点 (6) - 处理数量不固定的多点
                if  gt_info["match"] is not None:
                    cusps_abs = gt_info["match"].to(device).to(dtype)
                    cusps_rel = cusps_abs - pred_center_abs
                    # 计算 1536 个采样点到所有牙尖点的距离矩阵 [1536, num_cusps]
                    dist_to_cusps = torch.cdist(rel_pos, cusps_rel)

                    # 热图：取距离最近的那个牙尖点作为响应来源
                    min_dists, min_indices = dist_to_cusps.min(dim=1)
                    target_hm[:, 6] = torch.exp(-(min_dists ** 2) / (2 * self.sigma ** 2))

                    # 偏移：每个采样点回归向距离它最近的那个牙尖点
                    target_cusp_locs = cusps_rel[min_indices]
                    target_off[:, 6, :] = target_cusp_locs - rel_pos

                    # 牙尖点的回归掩码：针对每个 GT 牙尖点，采集其周围的 Top-K
                    for c_idx in range(cusps_rel.shape[0]):
                        c_dist = dist_to_cusps[:, c_idx]
                        k_real = min(self.k_offset, c_dist.shape[0])
                        _, topk_c_idx = torch.topk(c_dist, k=k_real, largest=False)
                        offset_mask[topk_c_idx, 6] = True

                # C. 计算损失
                stage2_hm_loss_total += F.mse_loss(pred_hm, target_hm)

                if offset_mask.any():
                    # 仅对靠近关键点的有效点进行 Offset 回归
                    stage2_off_loss_total += F.smooth_l1_loss(pred_off[offset_mask], target_off[offset_mask])

        # 均一化
        final_hm_loss = stage2_hm_loss_total / (B * max_teeth)
        final_off_loss = stage2_off_loss_total / max(valid_crops_count, 1)

        return final_hm_loss + final_off_loss, final_hm_loss, final_off_loss
