import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from functools import partial
import torch.nn.functional as F
import numpy as np


# --- 基础工具函数 ---
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x[:, :3, :], k=k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idxx = idx + idx_base
    idxx = idxx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idxx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    center = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - center, center), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature, idx


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=20, groups=8):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x, idx=None):
        identity = x
        x_graph, idx = get_graph_feature(x, k=self.k, idx=idx)
        x = self.conv(x_graph)
        x = x.max(dim=-1, keepdim=False)[0]
        x = x + self.shortcut(identity)
        return x, idx


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class LocalGeometricRefiner(nn.Module):
    def __init__(self, in_dim, out_dim, topk, groups=8):
        super().__init__()
        self.topk = topk
        # 处理拼接后的综合特征 (相对坐标 + 特征 + 距离 + 类别Emb)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 融合 Max Pool 和 Avg Pool 的特征
        self.mlp2 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, grouped_features):
        # grouped_features: (B, in_dim, C*TopK, K_reg)

        # 1. 局部逐点特征提取
        feat = self.mlp1(grouped_features)  # (B, out_dim, C*TopK, K_reg)

        # 2. 双路池化 (保留极值特征 + 平滑上下文)
        feat_max = torch.cat([feat[..., :-self.topk, :].max(dim=-1)[0], feat[..., -self.topk:, :32].max(dim=-1)[0]], dim=-1)  # (B, out_dim, C*TopK)
        feat_avg = torch.cat([feat[..., :-self.topk, :].mean(dim=-1), feat[..., -self.topk:, :32].mean(dim=-1)], dim=-1)  # (B, out_dim, C*TopK)

        # 3. 拼接并降维
        feat_pooled = torch.cat([feat_max, feat_avg], dim=1)  # (B, out_dim*2, C*TopK)

        # 转换维度给 Linear 层: (B, C*TopK, out_dim*2) -> (B, C*TopK, out_dim)
        feat_pooled = feat_pooled.permute(0, 2, 1).contiguous()
        out = self.mlp2(feat_pooled)

        return out
# --- 主模型 ---
class ToothLandmark(nn.Module):
    def __init__(self, classnums=6, k=20, topk=50):
        super(ToothLandmark, self).__init__()
        self.k = k
        self.topk = topk
        self.dim = 64
        self.classnums = classnums
        groups = 8

        # 1. 骨干网络
        self.edge_conv1 = EdgeConvBlock(3, self.dim, k=k, groups=groups)
        self.edge_conv2 = EdgeConvBlock(self.dim, self.dim * 2, k=k, groups=groups)
        self.edge_conv3 = EdgeConvBlock(self.dim * 2, self.dim * 4, k=k, groups=groups)
        self.edge_conv4 = EdgeConvBlock(self.dim * 4, self.dim * 4, k=k, groups=groups)

        # 2. 特征融合
        fusion_dim = self.dim * 11
        self.agg_conv1 = nn.Sequential(
            nn.Conv1d(fusion_dim, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.head_dim = self.dim * 4 + fusion_dim
        self.conv71 = nn.Sequential(
            nn.Conv1d(self.head_dim, self.dim * 16, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. 原始热图及分类分支
        self.heatmap_head = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, classnums, kernel_size=1))

        self.offset_head = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, classnums * 3, kernel_size=1))

        self.class_t = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, classnums, kernel_size=1))

        self.seg_head = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, 3, kernel_size=1))

        # 4. 直接坐标回归头
        self.k_reg = 128  # 建议稍微调大邻域点数，32 或 48，给 MLP 更多上下文

        self.pos_embed = nn.Sequential(
            nn.Conv2d(4, self.dim * 2, kernel_size=1),  # 输入改为4：(dx, dy, dz, distance)
            nn.GroupNorm(8, self.dim * 2),
            nn.LeakyReLU(0.2)
        )
        self.class_emb = nn.Embedding(self.classnums, self.dim * 2)

        self.mapliner = nn.Sequential(
            nn.Conv2d(self.dim * 16, self.dim * 4, kernel_size=1),
            nn.GroupNorm(8, self.dim * 4),
            nn.LeakyReLU(0.2)
        )

        # refiner 输入维度: 相对坐标(dim*2) + 降维特征(dim*4) + 类别Emb(dim*2) = dim*8
        refiner_in_dim = self.dim * 8
        self.local_refiner = LocalGeometricRefiner(in_dim=refiner_in_dim, out_dim=self.dim * 8, topk=self.topk)

        self.clsLiner = nn.Linear(1, self.dim * 8)
        self.predconf = nn.Sequential(
            nn.Linear(self.dim * 8, self.dim * 4),
            nn.LayerNorm(self.dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim * 4, 1))

        self.predland = nn.Sequential(
            nn.Linear(self.dim * 8, self.dim * 4),
            nn.LayerNorm(self.dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim * 4, 3))

    def forward(self, x):
        raw_pos = x.clone()
        x = x.permute(0, 2, 1)  # (B, 3, N)
        batch_size, _, num_points = x.size()

        # --- 特征提取 ---
        x1, idx = self.edge_conv1(x, idx=None)
        x2, idx = self.edge_conv2(x1, idx=idx)
        x3, idx = self.edge_conv3(x2, idx=idx)
        x4, idx = self.edge_conv4(x3, idx=idx)

        local_features = torch.cat((x1, x2, x3, x4), dim=1)
        combined = self.agg_conv1(local_features)
        global_feature = combined.max(dim=-1, keepdim=True)[0].repeat(1, 1, num_points)

        final_features = torch.cat((global_feature, local_features), dim=1)
        final_features = self.conv71(final_features)  # (B, 1024, N)

        # --- 第一阶段输出 ---
        heatmap = torch.sigmoid(self.heatmap_head(final_features))  # (B, C, N)
        cls_scores = torch.sigmoid(self.class_t(final_features)).permute(0, 2, 1)  # (B, N, C)
        out_seg = self.seg_head(final_features)

        offset_raw = self.offset_head(final_features).view(batch_size, self.classnums, 3, num_points)  # (B, C, 3, N)
        offset_return = offset_raw.permute(0, 3, 1, 2).contiguous()  # (B, N, C, 3)

        # --- 第二阶段：坐标回归分支 ---
        B, C, N = heatmap.shape
        K_edge = idx.shape[-1]

        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)
        neighbor_heatmap = torch.gather(heatmap.unsqueeze(-1).expand(-1, -1, -1, K_edge), 2, idx_expanded)
        local_max = heatmap >= neighbor_heatmap.max(dim=-1)[0]

        # 跨类别抑制 (Cross-Class Suppression)
        max_class_heatmap = heatmap.max(dim=1, keepdim=True)[0]
        class_margin = heatmap - max_class_heatmap
        temperature = 0.2
        cross_class_weight = torch.exp(class_margin / temperature)

        peak_heatmap = heatmap * local_max.float() * cross_class_weight

        # 1. 选取每类得分前 TopK 的点
        scores, topk_indices = torch.topk(peak_heatmap, k=self.topk, dim=-1)  # (B, C, TopK)
        flat_topk_idx = topk_indices.view(batch_size, -1)  # (B, C*TopK)
        num_candidates = flat_topk_idx.shape[1]

        # =========================================================================
        # --- [核心改进 1: 修正后的局部热图加权聚合 (Mean Shift in Center Space)] ---
        # =========================================================================
        # 为了避免影响第一阶段梯度，此处使用 detach
        offset_permuted = offset_raw.detach().permute(0, 1, 3, 2).contiguous()  # (B, C, N, 3)

        # 计算所有点依据预测offset偏移后的"目标聚集中心" (B, C, N, 3)
        # raw_pos 扩展为 (B, 1, N, 3) 以便加上类别相关的 offset
        shifted_pos_all = raw_pos.unsqueeze(1) + offset_permuted

        # 提取候选点(TopK)对应的"目标聚集中心" (B, C, TopK, 3)
        topk_idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        select_shifted_pos = torch.gather(shifted_pos_all, 2, topk_idx_expanded)

        # 1. 计算候选中心 与 原始点云经过偏移后的所有中心 的欧氏距离！
        # 这一步保证了我们在"聚集后的特征空间"中寻找邻居
        dist_per_class = torch.cdist(select_shifted_pos, shifted_pos_all)  # (B, C, TopK, N)
        dist_to_all = dist_per_class.view(batch_size, -1, num_points)  # (B, C*TopK, N)

        # 2. 获取在这个"聚集空间"中距离候选点最近的 k_reg 个近邻
        knn_dist_search, knn_idx_search = dist_to_all.topk(self.k_reg, dim=-1, largest=False)  # (B, C*TopK, k_reg)

        # 重塑维度
        knn_dist_search = knn_dist_search.view(batch_size, self.classnums, self.topk, self.k_reg)
        knn_idx_search_c = knn_idx_search.view(batch_size, self.classnums, self.topk * self.k_reg)

        # 3. 提取这些近邻点的信息：原始坐标、Offset、热图得分
        # (a) 提取坐标 (B, C, TopK, K_reg, 3)
        knn_idx_flat_search = knn_idx_search.view(batch_size, -1)
        knn_pos_search = torch.gather(raw_pos, 1, knn_idx_flat_search.unsqueeze(-1).expand(-1, -1, 3))
        knn_pos_search = knn_pos_search.view(batch_size, self.classnums, self.topk, self.k_reg, 3)

        # (b) 提取对应的 offset (B, C, TopK, K_reg, 3)
        knn_idx_search_c_exp = knn_idx_search_c.unsqueeze(-1).expand(-1, -1, -1, 3)
        knn_offset_search = torch.gather(offset_permuted, 2, knn_idx_search_c_exp)
        knn_offset_search = knn_offset_search.view(batch_size, self.classnums, self.topk, self.k_reg, 3)

        # ★ 获取这些近邻点 "各自预测的中心坐标"
        knn_pred_centers = knn_pos_search + knn_offset_search  # (B, C, TopK, K_reg, 3)

        # (c) 提取对应的 热图得分 (B, C, TopK, K_reg) (作为权重)
        knn_heatmap_search = torch.gather(heatmap.detach(), 2, knn_idx_search_c)
        knn_heatmap_search = knn_heatmap_search.view(batch_size, self.classnums, self.topk, self.k_reg)

        # 4. 应用距离阈值过滤偏远杂乱点
        # (注意：现在的 knn_dist_search 是"预测中心"之间的距离，<1 的约束更加合理有效)
        dist_threshold = 1.0
        dist_mask = (knn_dist_search < dist_threshold).float()

        # 5. 加权融合计算最终的 select_center_pos
        weights = knn_heatmap_search * dist_mask  # (B, C, TopK, K_reg)
        weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-6
        weights_norm = weights / weights_sum

        select_center_pos_c = torch.sum(weights_norm.unsqueeze(-1) * knn_pred_centers, dim=-2)  # (B, C, TopK, 3)

        # 展平以便与后续代码兼容
        select_center_pos = select_center_pos_c.view(batch_size, -1, 3)  # (B, C*TopK, 3)
        # =========================================================================

        # cls_idx = torch.arange(self.classnums, device=x.device).view(1, self.classnums, 1).expand(batch_size,
        #                                                                                           self.classnums,
        #                                                                                           self.topk)
        # cls_idx_flat = cls_idx.reshape(batch_size, -1)  # (B, C*TopK)
        # target_cls_emb = self.class_emb(cls_idx_flat)  # (B, C*TopK, dim)
        # target_cls_emb = target_cls_emb.unsqueeze(2).expand(-1, -1, self.k_reg, -1)  # (B, C*TopK, K_reg, dim)

        # --- [核心改进 2]: 在预测的中心点附近寻找邻居，而不是在表面点 ---
        dist = torch.cdist(select_center_pos, raw_pos)
        _, knn_idx_reg = dist.topk(self.k_reg, dim=-1, largest=False)
        knn_idx_reg_flat = knn_idx_reg.view(batch_size, -1)

        knn_pos = torch.gather(raw_pos, 1, knn_idx_reg_flat.unsqueeze(-1).expand(-1, -1, 3))
        knn_pos = knn_pos.view(batch_size, num_candidates, self.k_reg, 3)

        final_features_t = final_features.permute(0, 2, 1)
        knn_feat = torch.gather(final_features_t, 1, knn_idx_reg_flat.unsqueeze(-1).expand(-1, -1, self.dim * 16))
        knn_feat = knn_feat.view(batch_size, num_candidates, self.k_reg, self.dim * 16)

        heatmap_t = heatmap.permute(0, 2, 1).contiguous()
        #knn_heatmap = torch.gather(heatmap_t.detach(), 1, knn_idx_reg_flat.unsqueeze(-1).expand(-1, -1, self.classnums))
        #knn_heatmap = knn_heatmap.view(batch_size, num_candidates, self.k_reg, self.classnums)

        #knn_cls = torch.gather(cls_scores.detach(), 1, knn_idx_reg_flat.unsqueeze(-1).expand(-1, -1, self.classnums))
        #knn_cls = knn_cls.view(batch_size, num_candidates, self.k_reg, self.classnums)

        # 计算相对坐标：邻居点相对于“预测中心”的位置，为后续残差做准备
        rel_pos = knn_pos - select_center_pos.unsqueeze(2)  # (B, C*TopK, K_reg, 3)
        dist_pos = torch.norm(rel_pos, dim=-1, keepdim=True)  # (B, C*TopK, K_reg, 1)
        rel_pos_with_dist = torch.cat([rel_pos, dist_pos], dim=-1)  # (B, C*TopK, K_reg, 4)

        rel_pos_emb = self.pos_embed(rel_pos_with_dist.permute(0, 3, 1, 2).contiguous())  # (B, dim*2, C*TopK, K_reg)

        # 处理原有特征
        knn_feat_emb = self.mapliner(knn_feat.permute(0, 3, 1, 2).contiguous())  # (B, dim*4, C*TopK, K_reg)

        # 类别特征
        cls_idx = torch.arange(self.classnums, device=x.device).view(1, self.classnums, 1).expand(batch_size,
                                                                                                  self.classnums,
                                                                                                  self.topk)
        target_cls_emb = self.class_emb(cls_idx.reshape(batch_size, -1))  # (B, C*TopK, dim*2)
        target_cls_emb = target_cls_emb.unsqueeze(-1).expand(-1, -1, -1, self.k_reg).permute(0, 2, 1,
                                                                                             3).contiguous()  # (B, dim*2, C*TopK, K_reg)

        # 拼接特征送到专用的局部几何提取器
        grouped_features = torch.cat([knn_feat_emb, rel_pos_emb, target_cls_emb], dim=1)  # (B, dim*8, C*TopK, K_reg)

        # 使用提纯器替代之前的 MLP 和 Transformer
        reg_hidden = self.local_refiner(grouped_features)  # 输出: (B, C*TopK, dim*8)

        # 恢复形状 (B, C, TopK, dim*8)
        reg_hidden = reg_hidden.view(batch_size, self.classnums, self.topk, -1)

        # 5. 回归与置信度预测
        #cls = self.clsLiner(inverse_sigmoid(scores.unsqueeze(dim=-1)))
        reg_conf = self.predconf(reg_hidden)

        # pred_delta 预测残差
        pred_delta = self.predland(reg_hidden)

        # 最终预测 = 第一阶段预测中心 + 第二阶段微调残差
        select_center_pos_reshaped = select_center_pos.view(batch_size, self.classnums, self.topk, 3)
        final_pred_land = select_center_pos_reshaped + pred_delta

        reg_conf = scores.unsqueeze(dim=-1)
        return out_seg, heatmap_t, offset_return, cls_scores, final_pred_land, reg_conf, pred_delta, select_center_pos_reshaped

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
if __name__ == "__main__":

    model = ToothLandmark()#

    heatt_loss = ToothHeatmapLoss()
    off_loss = ToothOffsetLoss()
    match_loss = MultiMultiInstanceMatchLoss()
    criterion = nn.CrossEntropyLoss()


    out_seg, preheat_map, preoff_map, cls, pred_landmarks, reg_conf, pred_delta, select_pos = model(batch_data)
    dice_loss_ = heatt_loss(preheat_map, heat_map, mask)
    cls_loss_ = heatt_loss(cls, gtcls, gtcls.cuda().bool())
    seg_loss_ = criterion(out_seg, seg_label)
    # seg_loss_ = focalLoss(preheat_map, heat_map)
    # wing_Loss_ = wingLoss(preoff_map[mask], offest_map[mask])
    wing_Loss_ = off_loss(preoff_map.float(), offest_map.float(), mask)

    # coord_loss_, conf_loss_, combined_loss_ = match_loss(pred_landmarks, reg_conf, select_pos, teeth_landmarks, pred_delta)
    coord_loss_, conf_loss_, combined_loss_ = match_loss(pred_landmarks, reg_conf, teeth_landmarks, select_pos)

    loss = dice_loss_ + wing_Loss_ + cls_loss_ + combined_loss_ + seg_loss_

    scaler.scale(loss).backward()

    # Unscales gradients and calls
    scaler.step(optimizer)
    # Updates the scale for next iteration
    scaler.update()