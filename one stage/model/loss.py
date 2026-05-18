import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from config import config as cfg





def diceLoss(score, target):
    target = target.float()
    score =score.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def focalLoss(pred, gt):
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


# class WingLoss(nn.Module):
#     def __init__(self, omega=0.01, epsilon=2):
#         super(WingLoss, self).__init__()
#         self.omega = omega
#         self.epsilon = epsilon
#         self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
#     def forward(self, pred, target):
#         y = target
#         y_hat = pred
#         delta_2 = (y - y_hat).pow(2).sum(dim=-1, keepdim=False)
#         # delta = delta_2.sqrt()
#         delta = delta_2.clamp(min=1e-6).sqrt()
#
#         loss = torch.where(
#             delta < self.omega,
#             self.omega * torch.log(1 + delta / self.epsilon),
#             delta - self.C
#         )
#         return loss


class ToothHeatmapLoss(nn.Module):
    """
    适用于点云热图的连续型 Focal Loss。
    即使没有 GT=1 的点，也能通过高斯权重的分布进行回归。
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super(ToothHeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt, mask):
        """
        pred: (B, N, 6) - 经过 Sigmoid 的预测值
        gt:   (B, N, 6) - 高斯生成的热图标签
        """
        # 防止数值溢出
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        # 核心逻辑：利用 GT 的连续值进行加权
        # 当 GT 接近 1 时，第一项起主导作用；当 GT 接近 0 时，第二项（惩罚项）起主导作用
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred) * torch.pow(gt, self.beta)
        neg_loss = -torch.pow(pred, self.alpha) * torch.log(1 - pred) * torch.pow(1 - gt, self.beta)
        pos_loss, neg_loss = pos_loss[mask].mean(),  neg_loss[~mask].mean()

        loss = pos_loss + neg_loss
        return loss
class ToothOffsetLoss(nn.Module):
    """
    只针对有效关键点附近的点计算偏移量回归损失
    """

    def __init__(self, beta=1.0):
        super(ToothOffsetLoss, self).__init__()
        self.beta = beta  # SmoothL1 的阈值

    def forward(self, pred, gt, mask):
        """
        pred: (B, N, 6, 3)
        gt:   (B, N, 6, 3)
        mask: (B, N, 6) - 0或1，由 get_hotmap 函数生成
        """
        # 将 mask 扩展到 (B, N, 6, 3)
        mask = mask.unsqueeze(-1).expand_as(gt)

        # 只计算有效区域
        if mask.sum() > 0:
            # 仅提取 mask 为 1 的部分计算 Smooth L1
            loss = F.smooth_l1_loss(pred[mask == 1], gt[mask == 1], beta=self.beta, reduction='mean')
        else:
            loss = torch.tensor(0.0).to(pred.device)

        return loss

def clsfocal_loss(pred, gt):
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

class WingLoss(nn.Module):
    def __init__(self, omega=0.01, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
    def forward(self, pred, target, weights = None):
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
        if None != weights:
            weighted_loss = loss * weights
            return weighted_loss.sum() / (weights.sum() + 1e-6)
        else:
            return loss.mean()





def quality_focal_loss(pred_logits, target_scores, beta=2.0):
    """
    改进版 Quality Focal Loss
    pred_logits: (B, C, TopK, 1) 或 (B, C, TopK)
    target_scores: (B, C, TopK)
    """
    # 1. 维度对齐防范：消除最后的隐式维度 1
    if pred_logits.dim() > target_scores.dim():
        pred_logits = pred_logits.squeeze(-1)

    pred_sigmoid = torch.sigmoid(pred_logits)

    # 2. 预测值与目标值之间的绝对误差，作为 Focal 调节权重
    scale_factor = torch.abs(pred_sigmoid - target_scores)

    # 3. 计算无归约的 BCE Loss
    bce_loss = F.binary_cross_entropy_with_logits(
        pred_logits, target_scores, reduction='none'
    )

    # 4. QFL 核心公式
    loss = (scale_factor ** beta) * bce_loss

    # 5. 关键修正：按正样本的软标签权重求和进行归一化，而非无脑 mean()
    # 假设 target_scores 反映了该点作为正样本的质量
    pos_weight = target_scores.sum()

    # 防止除以 0，设定最小值为 1.0
    pos_weight = torch.clamp(pos_weight, min=1.0)

    # 对所有元素求和，再除以正样本总权重
    return loss.sum() / pos_weight


class MultiInstanceMatchLoss(nn.Module):
    def __init__(self, class_names=None):
        super(MultiInstanceMatchLoss, self).__init__()
        # 假设你的 WingLoss 实例已定义
        self.wingloss = WingLoss()
        self.class_names = class_names if class_names else [
            'Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'Cusp'
        ]

    # 修改 1：在前向传播中引入 select_pos_reshaped
    def forward(self, pred_landmarks, reg_conf, target_dict, select_pos_reshaped):

        B, C, K, _ = pred_landmarks.shape
        device = pred_landmarks.device

        # 转换形状方便处理
        pred_coords = pred_landmarks.float()
        select_coords = select_pos_reshaped.float()  # 第一阶段的初始候选点
        conf_logits = reg_conf.view(B, C, K).float()

        total_coord_loss = torch.tensor(0.0, device=device)
        total_conf_loss = torch.tensor(0.0, device=device)
        match_count = 0

        # 遍历 Batch
        for b in range(B):
            # 遍历类别
            for c_idx, c_name in enumerate(self.class_names):
                # 1. 获取该类在当前样本中的 GT
                if len(target_dict[c_name]) >= 1:
                    gt_pts = target_dict[c_name][b]  # (M, 3)
                    if gt_pts.dim() == 3: gt_pts = gt_pts.squeeze(0)
                    num_gt = gt_pts.shape[0] if gt_pts.ndim > 0 and gt_pts.numel() > 0 else 0
                else:
                    num_gt = 0

                curr_pred_coords = pred_coords[b, c_idx]  # (K, 3) 最终预测点
                curr_select_coords = select_coords[b, c_idx]  # (K, 3) 初始候选点
                curr_conf_logits = conf_logits[b, c_idx]  # (K,)

                # 2. 如果 GT 为空 (缺牙情况)
                if num_gt == 0:
                    conf_target = torch.zeros(K, device=device)
                    # 统一使用 quality_focal_loss，保持梯度计算方式一致
                    total_conf_loss += quality_focal_loss(curr_conf_logits, conf_target, beta=2.0)
                    continue

                gt_pts = gt_pts.to(device).float()

                # 3. 如果 GT 存在，进行匈牙利匹配
                with torch.no_grad():
                    # 修改 2：用初始候选点 (curr_select_coords) 计算匹配代价矩阵！！！
                    dist_matrix_match = torch.cdist(curr_select_coords, gt_pts, p=2)  # (K, M)

                    # 结合置信度作为代价
                    conf_matrix = 1.0 - torch.sigmoid(curr_conf_logits).unsqueeze(1).repeat(1, num_gt)

                    # 综合代价：初始距离越近、置信度越高，Cost 越小
                    cost_matrix = dist_matrix_match + 0.5 * conf_matrix

                    # 匹配：为每个 GT 找到最合适的初始候选点
                    pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().numpy())

                    # --- 新增：计算软标签 (Soft Labels) ---
                    # 修改 3：虽然用 select_coords 匹配，但评估“质量”时要用最终的 pred_coords
                    # 因为我们希望 conf 预测的是最终落点的精确度
                    final_dist_matrix = torch.cdist(curr_pred_coords, gt_pts, p=2)

                    # 提取成功匹配的点对之间最终的欧氏距离
                    matched_dists = final_dist_matrix[pred_indices, gt_indices]  # (num_matches,)

                    # 将距离转换为 0~1 的软标签分数 (使用高斯衰减)
                    sigma = 2.0  # 根据你的坐标系尺度调整
                    soft_scores = torch.exp(-(matched_dists ** 2) / (2 * sigma ** 2)).detach()

                # 4. 计算置信度损失 (软标签赋值)
                conf_target = torch.zeros(K, device=device)
                conf_target[pred_indices] = soft_scores

                # 送入改进版的 quality_focal_loss
                total_conf_loss += quality_focal_loss(curr_conf_logits, conf_target, beta=2.0)

                # 5. 计算坐标损失 (只对匹配上的对计算，计算最终预测点与GT的误差)
                matched_preds = curr_pred_coords[pred_indices]
                matched_gts = gt_pts[gt_indices]

                total_coord_loss += self.wingloss(matched_preds, matched_gts, weights = None)
                match_count += 1

        # 归一化损失
        avg_conf_loss = total_conf_loss / (B * len(self.class_names))
        avg_coord_loss = total_coord_loss / max(match_count, 1)

        # 最终综合 Loss
        combined_loss = avg_coord_loss + avg_conf_loss

        return avg_coord_loss, avg_conf_loss, combined_loss



class MultiInstanceMatchLoss1(nn.Module):
    def __init__(self, class_names=None, sigma=3.0, unmatched_weight=0.1):
        super(MultiInstanceMatchLoss1, self).__init__()
        self.wingloss = WingLoss()  # 假设已定义
        self.sigma = sigma
        self.unmatched_weight = unmatched_weight  # 未匹配点的惩罚权重
        self.class_names = class_names if class_names else [
            'Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'Cusp'
        ]

    def forward(self, pred_landmarks, reg_conf, select_pos_reshaped, target_dict, pred_delta):
        """
        新增输入:
        pred_delta: 第二阶段网络直接输出的偏移量 (B, C, TopK, 3)，未加初始坐标前
        """
        B, C, K, _ = pred_landmarks.shape
        device = pred_landmarks.device

        pred_coords = pred_landmarks.float()
        select_coords = select_pos_reshaped.float()
        conf_logits = reg_conf.view(B, C, K).float()
        # 这里的 pred_delta 是为了对未匹配点做“归零”约束
        curr_deltas = pred_delta.view(B, C, K, 3).float()

        total_coord_loss = torch.tensor(0.0, device=device)
        total_conf_loss = torch.tensor(0.0, device=device)
        match_count = 0

        for b in range(B):
            for c_idx, c_name in enumerate(self.class_names):
                # 1. 获取 GT
                if len(target_dict[c_name]) >= 1:
                    gt_pts = target_dict[c_name][b]
                    if gt_pts.dim() == 3: gt_pts = gt_pts.squeeze(0)
                    num_gt = gt_pts.shape[0] if gt_pts.ndim > 0 and gt_pts.numel() > 0 else 0
                else:
                    num_gt = 0

                c_pred_coords = pred_coords[b, c_idx]  # (K, 3)
                c_select_coords = select_coords[b, c_idx]  # (K, 3)
                c_conf_logits = conf_logits[b, c_idx]  # (K,)
                c_deltas = curr_deltas[b, c_idx]  # (K, 3)

                # 2. 空 GT 处理 (全设为背景)
                if num_gt == 0:
                    conf_target = torch.zeros(K, device=device)
                    total_conf_loss += quality_focal_loss(c_conf_logits, conf_target)
                    # 约束：所有点偏移回归 0
                    total_coord_loss += self.unmatched_weight * F.mse_loss(c_deltas, torch.zeros_like(c_deltas))
                    continue

                gt_pts = gt_pts.to(device).float()

                # 3. 匈牙利匹配 (基于初始点 select_coords)
                with torch.no_grad():
                    dist_matrix_match = torch.cdist(c_select_coords, gt_pts, p=2)
                    conf_matrix = 1.0 - torch.sigmoid(c_conf_logits).unsqueeze(1).repeat(1, num_gt)
                    cost_matrix = dist_matrix_match + 0.5 * conf_matrix
                    pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().numpy())

                    # 计算匹配成功的 Soft Labels (基于最终点 pred_coords)
                    final_dist_matrix = torch.cdist(c_pred_coords, gt_pts, p=2)
                    matched_dists = final_dist_matrix[pred_indices, gt_indices]
                    soft_scores = torch.exp(-(matched_dists ** 2) / (2 * self.sigma ** 2)).detach()

                # 4. 置信度损失 (QFL)
                conf_target = torch.zeros(K, device=device)
                conf_target[pred_indices] = soft_scores
                total_conf_loss += quality_focal_loss(c_conf_logits, conf_target)

                # 5. 坐标损失
                # A: 正样本回归 (WingLoss)
                matched_preds = c_pred_coords[pred_indices]
                matched_gts = gt_pts[gt_indices]
                total_coord_loss += self.wingloss(matched_preds, matched_gts)

                # B: 背景点“归零”约束 (防止悬空)
                # 找出所有没被匹配上的索引
                all_indices = set(range(K))
                matched_set = set(pred_indices)
                unmatched_indices = list(all_indices - matched_set)

                if len(unmatched_indices) > 0:
                    unmatched_deltas = c_deltas[unmatched_indices]
                    # 强制这些点的预测偏移为 0，即留在一阶段 Heatmap 选出的位置
                    total_coord_loss += self.unmatched_weight * F.mse_loss(unmatched_deltas,
                                                                           torch.zeros_like(unmatched_deltas))

                match_count += 1

        avg_conf_loss = total_conf_loss / (B * len(self.class_names))
        avg_coord_loss = total_coord_loss / max(match_count, 1)
        combined_loss = avg_coord_loss + avg_conf_loss

        return avg_coord_loss, avg_conf_loss, combined_loss



class MultiInstanceMatchLoss2(nn.Module):
    def __init__(self, class_names=None, sigma=3.0, unmatched_weight=0.1):
        super(MultiInstanceMatchLoss2, self).__init__()
        # self.wingloss = WingLoss()  # 假设已在外部定义
        self.sigma = sigma
        self.unmatched_weight = unmatched_weight
        self.class_names = class_names if class_names else [
            'Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'Cusp'
        ]
        self.wingloss = WingLoss()  # 假设已定义
    def forward(self, pred_landmarks, reg_conf, select_pos_reshaped, target_dict, pred_delta):
        B, C, K, _ = pred_landmarks.shape
        device = pred_landmarks.device

        pred_coords = pred_landmarks.float()
        select_coords = select_pos_reshaped.float()
        conf_logits = reg_conf.view(B, C, K).float()
        curr_deltas = pred_delta.view(B, C, K, 3).float()

        total_coord_loss = torch.tensor(0.0, device=device)
        total_matched_pts = 0  # 记录整个 batch 中成功匹配的总点数
        total_unmatched_pts = 0  # 记录未匹配的点数用于对齐量级

        # 用于收集整个 batch 的 QFL 输入，实现全局归一化
        all_conf_logits = []
        all_conf_targets = []

        for b in range(B):
            for c_idx, c_name in enumerate(self.class_names):
                # 1. 获取 GT
                if len(target_dict[c_name]) >= 1:
                    gt_pts = target_dict[c_name][b]
                    if gt_pts.dim() == 3: gt_pts = gt_pts.squeeze(0)
                    num_gt = gt_pts.shape[0] if gt_pts.ndim > 0 and gt_pts.numel() > 0 else 0
                else:
                    num_gt = 0

                c_pred_coords = pred_coords[b, c_idx]  # (K, 3)
                c_select_coords = select_coords[b, c_idx]  # (K, 3)
                c_conf_logits = conf_logits[b, c_idx]  # (K,)
                c_deltas = curr_deltas[b, c_idx]  # (K, 3)

                # 2. 空 GT 处理 (全设为背景)
                if num_gt == 0:
                    conf_target = torch.zeros(K, device=device)
                    all_conf_logits.append(c_conf_logits)
                    all_conf_targets.append(conf_target)

                    # 约束：所有点偏移回归 0
                    total_coord_loss += self.unmatched_weight * F.mse_loss(c_deltas, torch.zeros_like(c_deltas),
                                                                           reduction='sum')
                    total_unmatched_pts += K
                    continue

                gt_pts = gt_pts.to(device).float()

                # 3. 匈牙利匹配 (基于初始点 select_coords)
                with torch.no_grad():
                    dist_matrix_match = torch.cdist(c_select_coords, gt_pts, p=2)
                    conf_matrix = 1.0 - torch.sigmoid(c_conf_logits).unsqueeze(1).repeat(1, num_gt)
                    cost_matrix = dist_matrix_match + 0.5 * conf_matrix
                    pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().numpy())

                    # 计算匹配成功的 Soft Labels (基于最终点 pred_coords)
                    final_dist_matrix = torch.cdist(c_pred_coords, gt_pts, p=2)
                    matched_dists = final_dist_matrix[pred_indices, gt_indices]
                    # ★ 核心修正：必须 detach，防止 QFL 干扰坐标回归！
                    soft_scores = torch.exp(-(matched_dists ** 2) / (2 * self.sigma ** 2)).detach()

                # 4. 收集置信度目标
                conf_target = torch.zeros(K, device=device)
                conf_target[pred_indices] = soft_scores

                all_conf_logits.append(c_conf_logits)
                all_conf_targets.append(conf_target)

                # 5. 坐标损失计算
                # A: 正样本回归 (WingLoss)
                matched_preds = c_pred_coords[pred_indices]
                matched_gts = gt_pts[gt_indices]
                # 假设 WingLoss 默认 reduction='mean'，为了精确计算总误差，我们先拿到均值再乘回来，或者外部定义 reduction='sum'
                # 推荐你的 WingLoss 使用 reduction='sum'，这里按标准求和处理：
                total_coord_loss += self.wingloss(matched_preds, matched_gts) * len(matched_preds)
                total_matched_pts += len(matched_preds)

                # B: 背景点“归零”约束
                all_indices = set(range(K))
                matched_set = set(pred_indices)
                unmatched_indices = list(all_indices - matched_set)

                if len(unmatched_indices) > 0:
                    unmatched_deltas = c_deltas[unmatched_indices]
                    total_coord_loss += self.unmatched_weight * F.mse_loss(unmatched_deltas,
                                                                           torch.zeros_like(unmatched_deltas),
                                                                           reduction='sum')
                    total_unmatched_pts += len(unmatched_indices)

        # 6. 全局计算置信度损失 (QFL)
        # 将所有 logits 和 targets 拼接为 1D 张量，利用你写好的 QFL 进行一次性全局计算
        cat_conf_logits = torch.cat(all_conf_logits)  # (B * C * K)
        cat_conf_targets = torch.cat(all_conf_targets)  # (B * C * K)

        avg_conf_loss = quality_focal_loss(cat_conf_logits, cat_conf_targets)

        # 7. 全局计算坐标损失平均值
        # 为避免除以0，确保至少为1；同时把 matched 和 unmatched 产生的总点数作为分母，让 loss 尺度平稳
        total_pts = total_matched_pts + total_unmatched_pts
        avg_coord_loss = total_coord_loss / max(total_pts, 1.0)

        # 这里你可以根据需要调整两者的权重比例
        combined_loss = avg_coord_loss + avg_conf_loss

        return avg_coord_loss, avg_conf_loss, combined_loss



class MultiMultiInstanceMatchLoss(nn.Module):
    def __init__(self, class_names=None, sigma=2):#2.0
        super(MultiMultiInstanceMatchLoss, self).__init__()
        self.sigma = sigma
        self.class_names = class_names if class_names else [
            'Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'Cusp'
        ]
        # 假设 WingLoss 已经在外部定义，并且支持 reduction='none' 或 'mean'

        self.wingloss = WingLoss()
        self.num = {'Mesial':50, 'Distal':50, 'InnerPoint':50, 'OuterPoint':50, 'FacialPoint':50, 'Cusp':50}

    def forward(self, pred_landmarks, reg_conf, target_dict, select_pos):
        """
        Args:
            pred_landmarks: (B, C, K, 3) 第二阶段回归后的坐标
            reg_conf: (B, C, K, 1) 置信度 logits
            target_dict: 字典，key 为类别名，value 为长度为 B 的 list，每个元素为 (N_gt, 3) 的 Tensor
        """
        B, C, K, _ = pred_landmarks.shape
        device = pred_landmarks.device

        total_coord_loss = 0.0
        all_conf_logits = []
        all_conf_targets = []

        valid_element_count = 0

        for b in range(B):
            for c_idx, c_name in enumerate(self.class_names):
                # 1. 获取当前样本、当前类别的预测值
                select_p = select_pos[b, c_idx][:self.num[c_name], ...]
                curr_preds = pred_landmarks[b, c_idx][:self.num[c_name], ...]  # (K, 3)
                curr_conf_logits = reg_conf[b, c_idx].squeeze(-1)[:self.num[c_name], ...]  # (K,)

                # --- 修复 IndexError 的核心逻辑 ---
                # 检查类别是否存在于字典中，且该 sample 索引是否在列表范围内
                gt_pts = None
                if c_name in target_dict and b < len(target_dict[c_name]):
                    gt_pts = target_dict[c_name][b]

                # 确保 gt_pts 是有效的 Tensor 且不为空
                is_empty_gt = (
                        gt_pts is None or
                        (isinstance(gt_pts, torch.Tensor) and gt_pts.numel() == 0) or
                        (isinstance(gt_pts, list) and len(gt_pts) == 0)
                )

                # 2. 如果没有 GT (负样本处理)
                if is_empty_gt:
                    # 目标置信度全设为 0
                    all_conf_logits.append(curr_conf_logits)
                    all_conf_targets.append(torch.zeros(K, device=device))
                    # 此时不累加 total_coord_loss，也不增加 valid_element_count
                    continue
                # ----------------------------------

                # 3. 正常计算 (有 GT)

                gt_pts = torch.as_tensor(gt_pts, device=device, dtype=torch.float32)

                if gt_pts.dim() == 3:
                    gt_pts = gt_pts.squeeze(0)

                gt_pts = gt_pts.to(device)

                # 计算最近邻匹配
                dist_matrix = torch.cdist(select_p, gt_pts, p=2)
                min_dists, min_indices = torch.min(dist_matrix, dim=1)
                nearest_gts = gt_pts[min_indices]


                # 质量得分 (Quality Target)
                with torch.no_grad():
                    soft_label = torch.exp(-(min_dists ** 2) / (2 * self.sigma ** 2))
                soft_label_ = soft_label.clone()
                if c_name == 'Mesial':
                    soft_label_ = soft_label_*2
                # 坐标损失
                coord_loss_per_point = self.wingloss(curr_preds, nearest_gts, soft_label_)
                total_coord_loss += coord_loss_per_point.sum()
                valid_element_count += 10



                all_conf_logits.append(curr_conf_logits)
                all_conf_targets.append(soft_label)

        # 7. 全局损失归一化
        # 拼接所有置信度
        cat_logits = torch.cat(all_conf_logits)  # (B*C*K)
        cat_targets = torch.cat(all_conf_targets)  # (B*C*K)

        # 计算 Quality Focal Loss (QFL)
        avg_conf_loss = self.quality_focal_loss(cat_logits, cat_targets)

        # 计算平均坐标损失
        avg_coord_loss = total_coord_loss / max(valid_element_count, 1)

        # 最终加权损失 (可根据量级调整权重)
        combined_loss = avg_coord_loss + 2.0 * avg_conf_loss

        return avg_coord_loss, avg_conf_loss, combined_loss

    def quality_focal_loss(self, pred_logits, target_scores, beta=2.0):
        """
        Quality Focal Loss 适用于连续的 [0, 1] 目标
        """
        pred_sigmoid = torch.sigmoid(pred_logits)
        # 防止数值溢出
        pred_sigmoid = torch.clamp(pred_sigmoid, 1e-6, 1.0 - 1e-6)

        loss = - (target_scores * torch.log(pred_sigmoid) +
                  (1 - target_scores) * torch.log(1 - pred_sigmoid))

        # 权重因子 |target - pred|^beta
        weight = torch.pow(torch.abs(target_scores - pred_sigmoid), beta)

        return (loss * weight).mean()



class MultiMultiInstanceMatchLoss2(nn.Module):
    def __init__(self, class_names=None, sigma=2.0, pos_radius=3.0):
        """
        Args:
            sigma: 用于计算软标签的方差参数
            pos_radius: 正样本匹配的物理距离阈值 (例如 3.0 mm)。
                        只有距离 GT 在该半径内的预测点，才计算坐标 Loss。
        """
        super(MultiMultiInstanceMatchLoss2, self).__init__()
        self.sigma = sigma
        self.pos_radius = pos_radius
        self.class_names = class_names if class_names else [
            'Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'Cusp'
        ]

        # 假设 WingLoss 已经定义，签名类似: forward(pred, target, weight=None)
        # 注意：你需要确保你的 WingLoss 支持传入 weight 并能够正确加权求和
        self.wingloss = WingLoss()

    def forward(self, pred_landmarks, reg_conf, target_dict, select_pos):
        B, C, K, _ = pred_landmarks.shape
        device = pred_landmarks.device

        total_coord_loss = 0.0
        all_conf_logits = []
        all_conf_targets = []

        valid_element_count = 0.0

        for b in range(B):
            for c_idx, c_name in enumerate(self.class_names):
                select_p = select_pos[b, c_idx]  # (K, 3) 初始锚点
                curr_preds = pred_landmarks[b, c_idx]  # (K, 3) 微调后的最终坐标
                curr_conf_logits = reg_conf[b, c_idx].squeeze(-1)  # (K,)

                # 1. 提取 GT
                gt_pts = None
                if c_name in target_dict and b < len(target_dict[c_name]):
                    gt_pts = target_dict[c_name][b]

                is_empty_gt = (
                        gt_pts is None or
                        (isinstance(gt_pts, torch.Tensor) and gt_pts.numel() == 0) or
                        (isinstance(gt_pts, list) and len(gt_pts) == 0)
                )

                # 2. 负样本情况 (这颗牙齿没有这个类别的关键点)
                if is_empty_gt:
                    all_conf_logits.append(curr_conf_logits)
                    all_conf_targets.append(torch.zeros(K, device=device))  # 目标全设为 0
                    continue

                # 3. 正样本情况分析
                gt_pts = torch.as_tensor(gt_pts, device=device, dtype=torch.float32)
                if gt_pts.dim() == 3:
                    gt_pts = gt_pts.squeeze(0)

                # --- 核心改进：基于初始锚点进行匹配与正负样本截断 ---
                dist_matrix = torch.cdist(select_p, gt_pts, p=2)  # (K, N_gt)
                min_dists, min_indices = torch.min(dist_matrix, dim=1)  # (K,)
                nearest_gts = gt_pts[min_indices]  # (K, 3)

                # 确定正样本掩码 (只有初始点在 GT 附近的，才要求网络去微调坐标)
                pos_mask = min_dists < self.pos_radius

                # 计算置信度的 Soft Label (只对正样本计算 >0 的标签，负样本直接为 0)
                # 这非常关键，它教会网络：距离远的点，你的预测置信度必须降为 0！
                target_conf = torch.zeros(K, device=device)
                if pos_mask.any():
                    target_conf[pos_mask] = torch.exp(-(min_dists[pos_mask] ** 2) / (2 * self.sigma ** 2))

                all_conf_logits.append(curr_conf_logits)
                all_conf_targets.append(target_conf)

                # --- 坐标回归只监督正样本 ---
                if pos_mask.any():
                    pos_preds = curr_preds[pos_mask]  # (N_pos, 3)
                    pos_gts = nearest_gts[pos_mask]  # (N_pos, 3)
                    pos_weights = target_conf[pos_mask]  # (N_pos,) 使用质量分数作为回归权重

                    # 修复之前的越界 Bug：权重应用在 Loss 上，而不是 Target 上！
                    cls_weight_factor = 2.0 if c_name == 'Mesial' else 1.0
                    pos_weights = pos_weights * cls_weight_factor

                    # 计算坐标损失
                    # 确保你的 WingLoss 实现中，能处理传入的 pos_weights
                    coord_loss_per_point = self.wingloss(pos_preds, pos_gts, pos_weights)

                    total_coord_loss += coord_loss_per_point.sum()
                    valid_element_count += pos_mask.sum().item()

        # 4. 全局损失合并计算
        cat_logits = torch.cat(all_conf_logits)  # (B*C*K)
        cat_targets = torch.cat(all_conf_targets)  # (B*C*K)

        # 计算置信度损失 (由于负样本占比大，QFL 自带的难易样本挖掘非常有效)
        avg_conf_loss = self.quality_focal_loss(cat_logits, cat_targets)

        # 计算平均坐标损失 (避免除以 0)
        avg_coord_loss = total_coord_loss / max(valid_element_count, 1.0)

        # 最终损失融合
        combined_loss = avg_coord_loss + 2.0 * avg_conf_loss

        return avg_coord_loss, avg_conf_loss, combined_loss

    def quality_focal_loss(self, pred_logits, target_scores, beta=2.0):
        pred_sigmoid = torch.sigmoid(pred_logits)
        pred_sigmoid = torch.clamp(pred_sigmoid, 1e-6, 1.0 - 1e-6)

        # target_scores 现在严格在 [0, 1] 之间，不会导致 log(负数) 的错误
        loss = - (target_scores * torch.log(pred_sigmoid) +
                  (1 - target_scores) * torch.log(1 - pred_sigmoid))

        weight = torch.pow(torch.abs(target_scores - pred_sigmoid), beta)

        return (loss * weight).mean()


