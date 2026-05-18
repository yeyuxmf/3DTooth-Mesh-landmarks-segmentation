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

        # 4. 直接坐标回归头
        self.k_reg = 16 * 2
        self.pos_embed = nn.Sequential(
            nn.Conv2d(3, self.dim * 4, kernel_size=1),
            nn.GroupNorm(8, self.dim * 4),
            nn.LeakyReLU(0.2)
        )
        self.score_embed = nn.Sequential(
            nn.Conv2d(self.classnums, self.dim * 2, kernel_size=1),
            nn.GroupNorm(8, self.dim * 2),
            nn.LeakyReLU(0.2)
        )
        self.class_emb = nn.Embedding(self.classnums, self.dim * 2)

        reg_in_dim = self.dim * 14
        self.local_mlp = nn.Sequential(
            nn.Conv2d(reg_in_dim, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim * 8, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mapliner = nn.Linear(self.dim * 16, self.dim * 8)

        # 注意力权重网络：用于替代 MaxPool，模拟 NMS 中的加权质心
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(reg_in_dim, self.dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim, 1, kernel_size=1)
        )
        norm_layer = partial(RMSNorm, eps=1e-6)
        self.Tencoders = nn.ModuleList([
            Block(dim=self.dim * 8, num_heads=self.dim * 8 // 64, mlp_ratio=2, qkv_bias=True, norm_layer=norm_layer)
            for i in range(4)])

        self.clsLiner = nn.Linear(1, self.dim * 8)
        self.predconf = nn.Sequential(
            nn.Linear(self.dim * 8, self.dim * 4),
            nn.LayerNorm(self.dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim * 4, self.dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim * 4, 1))

        self.predland = nn.Sequential(
            nn.Linear(self.dim * 8, self.dim * 4),
            nn.LayerNorm(self.dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim * 4, self.dim * 4),
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

        offset_raw = self.offset_head(final_features).view(batch_size, self.classnums, 3, num_points)  # (B, C, 3, N)
        offset_return = offset_raw.permute(0, 3, 1, 2).contiguous()  # (B, N, C, 3) 返回给外部计算Loss使用

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

        # 提取候选点的原始表面坐标: (B, C*TopK, 3)
        select_raw_pos = torch.gather(raw_pos, 1, flat_topk_idx.unsqueeze(-1).expand(-1, -1, 3))

        # =========================================================================
        # --- [核心改进 1: 局部热图加权聚合 (Local Heatmap-Weighted Mean Shift)] ---
        # =========================================================================
        # 1. 计算所有候选点 (TopK) 与原始点云中所有点的欧氏距离
        dist_to_all = torch.cdist(select_raw_pos, raw_pos)  # (B, C*TopK, N)

        # 2. 获取候选点在原始表面上的 K 个近邻 (复用 k_reg 参数，或自定义邻域数)
        # 注意：这里是在寻找 "距离候选点最近的表面点"
        knn_dist_search, knn_idx_search = dist_to_all.topk(self.k_reg, dim=-1, largest=False)  # (B, C*TopK, k_reg)

        # 重塑维度以按类别处理
        knn_dist_search = knn_dist_search.view(batch_size, self.classnums, self.topk, self.k_reg)
        knn_idx_search_c = knn_idx_search.view(batch_size, self.classnums, self.topk * self.k_reg)

        # 3. 提取这些近邻点的信息：原始坐标、Offset、热图得分
        # (a) 提取坐标 (B, C, TopK, K_reg, 3)
        knn_idx_flat_search = knn_idx_search.view(batch_size, -1)
        knn_pos_search = torch.gather(raw_pos, 1, knn_idx_flat_search.unsqueeze(-1).expand(-1, -1, 3))
        knn_pos_search = knn_pos_search.view(batch_size, self.classnums, self.topk, self.k_reg, 3)

        # (b) 提取对应的 offset (B, C, TopK, K_reg, 3) (使用 detach 避免复杂梯度)
        offset_raw_permuted = offset_raw.detach().permute(0, 1, 3, 2).contiguous()  # (B, C, N, 3)
        knn_idx_search_c_exp = knn_idx_search_c.unsqueeze(-1).expand(-1, -1, -1, 3)
        knn_offset_search = torch.gather(offset_raw_permuted, 2, knn_idx_search_c_exp)
        knn_offset_search = knn_offset_search.view(batch_size, self.classnums, self.topk, self.k_reg, 3)

        # ★ 获取这些近邻点 "各自预测的中心坐标"
        knn_pred_centers = knn_pos_search + knn_offset_search  # (B, C, TopK, K_reg, 3)

        # (c) 提取对应的 热图得分 (B, C, TopK, K_reg) (作为权重)
        knn_heatmap_search = torch.gather(heatmap.detach(), 2, knn_idx_search_c)
        knn_heatmap_search = knn_heatmap_search.view(batch_size, self.classnums, self.topk, self.k_reg)

        # 4. 应用距离阈值 (< 0.7) 掩码过滤远点
        dist_threshold = 1
        dist_mask = (knn_dist_search < dist_threshold).float()

        # 5. 加权融合计算最终的 select_center_pos
        weights = knn_heatmap_search * dist_mask  # (B, C, TopK, K_reg)
        weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-6  # 加上极小值防止除以0
        weights_norm = weights / weights_sum  # 归一化权重

        # 将每个近邻预测的中心，按热图权重进行加权累加，得到更鲁棒的中心点！
        select_center_pos_c = torch.sum(weights_norm.unsqueeze(-1) * knn_pred_centers, dim=-2)  # (B, C, TopK, 3)

        # 展平以便与后续代码兼容
        select_center_pos = select_center_pos_c.view(batch_size, -1, 3)  # (B, C*TopK, 3)
        # =========================================================================

        cls_idx = torch.arange(self.classnums, device=x.device).view(1, self.classnums, 1).expand(batch_size,
                                                                                                  self.classnums,
                                                                                                  self.topk)
        cls_idx_flat = cls_idx.reshape(batch_size, -1)  # (B, C*TopK)
        target_cls_emb = self.class_emb(cls_idx_flat)  # (B, C*TopK, dim)
        target_cls_emb = target_cls_emb.unsqueeze(2).expand(-1, -1, self.k_reg, -1)  # (B, C*TopK, K_reg, dim)

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
        knn_heatmap = torch.gather(heatmap_t.detach(), 1, knn_idx_reg_flat.unsqueeze(-1).expand(-1, -1, self.classnums))
        knn_heatmap = knn_heatmap.view(batch_size, num_candidates, self.k_reg, self.classnums)

        knn_cls = torch.gather(cls_scores.detach(), 1, knn_idx_reg_flat.unsqueeze(-1).expand(-1, -1, self.classnums))
        knn_cls = knn_cls.view(batch_size, num_candidates, self.k_reg, self.classnums)

        # 计算相对坐标：邻居点相对于“预测中心”的位置，为后续残差做准备
        rel_pos = knn_pos - select_center_pos.unsqueeze(2)
        rel_pos = self.pos_embed(rel_pos.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        knn_cls_embed = self.score_embed(knn_heatmap.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        knn_feat = self.mapliner(knn_feat)

        grouped_features = torch.cat([knn_feat, rel_pos, target_cls_emb], dim=-1)
        grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()  # (B, Reg_dim, C*TopK, K_reg)

        # --- [核心改进 3]: Softmax Attention 加权融合，替换原来的 Max Pool ---
        attn_logits = self.attn_mlp(grouped_features)  # (B, 1, C*TopK, K_reg)
        attn_weights = F.softmax(attn_logits, dim=-1)  # 对 K_reg 维度做 softmax

        agg_feat_unpooled = self.local_mlp(grouped_features)  # (B, dim*8, C*TopK, K_reg)

        # 逐元素相乘后求和，模拟 NMS 里面的分数加权聚合
        agg_feat = torch.sum(agg_feat_unpooled * attn_weights, dim=-1)  # (B, dim*8, C*TopK)

        reg_hidden = agg_feat.permute(0, 2, 1).contiguous()
        reg_hidden = reg_hidden.view(batch_size, self.classnums, self.topk, -1)

        # 5. 回归与置信度预测
        cls = self.clsLiner(inverse_sigmoid(scores.unsqueeze(dim=-1)))
        reg_conf = self.predconf(reg_hidden)

        # pred_delta 现在预测的是基于第一阶段中心点的“微调残差”
        pred_delta = self.predland(reg_hidden)

        # --- [核心改进 4]: 最终预测 = 第一阶段预测中心 + 第二阶段微调残差 ---
        select_center_pos_reshaped = select_center_pos.view(batch_size, self.classnums, self.topk, 3)
        final_pred_land = select_center_pos_reshaped + pred_delta

        return heatmap_t, offset_return, cls_scores, final_pred_land, reg_conf, pred_delta, select_center_pos_reshaped