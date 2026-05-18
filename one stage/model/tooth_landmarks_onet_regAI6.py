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
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, grouped_features):
        feat = self.mlp1(grouped_features)
        feat_max = torch.cat([feat[..., :-self.topk, :].max(dim=-1)[0], feat[..., -self.topk:, :32].max(dim=-1)[0]],
                             dim=-1)
        feat_avg = torch.cat([feat[..., :-self.topk, :].mean(dim=-1), feat[..., -self.topk:, :32].mean(dim=-1)], dim=-1)
        feat_pooled = torch.cat([feat_max, feat_avg], dim=1)
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

        # ★ 关键修改 1：内部回归类数。对外宣称是 classnums=6，但网络预测头输出 out_c=7
        self.classnums = classnums
        self.out_c = classnums + 1  # 0和1为左右Mesial，共7个通道
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

        # 3. 原始热图及分类分支 (★ 头输出维度改为 self.out_c = 7)
        self.heatmap_head7= nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, self.out_c, kernel_size=1))  # 输出 7

        self.offset_head7 = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, self.out_c * 3, kernel_size=1))  # 输出 21

        self.class_t7 = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, self.out_c, kernel_size=1))  # 输出 7

        self.seg_head = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, 3, kernel_size=1))

        # 4. 直接坐标回归头 (保持 6 类不变，因为我们将会在选取候选点前把通道合并回来)
        self.k_reg = 128

        self.pos_embed = nn.Sequential(
            nn.Conv2d(4, self.dim * 2, kernel_size=1),
            nn.GroupNorm(8, self.dim * 2),
            nn.LeakyReLU(0.2)
        )
        self.class_emb = nn.Embedding(self.classnums, self.dim * 2)  # 仍是 6 种语义类

        self.mapliner = nn.Sequential(
            nn.Conv2d(self.dim * 16, self.dim * 4, kernel_size=1),
            nn.GroupNorm(8, self.dim * 4),
            nn.LeakyReLU(0.2)
        )

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

        x1, idx = self.edge_conv1(x, idx=None)
        x2, idx = self.edge_conv2(x1, idx=idx)
        x3, idx = self.edge_conv3(x2, idx=idx)
        x4, idx = self.edge_conv4(x3, idx=idx)

        local_features = torch.cat((x1, x2, x3, x4), dim=1)
        combined = self.agg_conv1(local_features)
        global_feature = combined.max(dim=-1, keepdim=True)[0].repeat(1, 1, num_points)

        final_features = torch.cat((global_feature, local_features), dim=1)
        final_features = self.conv71(final_features)

        # --- 第一阶段输出 (★ 这里全是 7 通道) ---
        heatmap_7 = torch.sigmoid(self.heatmap_head7(final_features))  # (B, 7, N)
        cls_scores_7 = torch.sigmoid(self.class_t7(final_features)).permute(0, 2, 1)  # (B, N, 7)
        out_seg = self.seg_head(final_features)

        offset_raw_7 = self.offset_head7(final_features).view(batch_size, self.out_c, 3, num_points)  # (B, 7, 3, N)
        offset_return_7 = offset_raw_7.permute(0, 3, 1, 2).contiguous()  # (B, N, 7, 3) 供 Loss 使用

        # --- 局部非极大值抑制 (NMS) 在 7 通道下独立进行，完美避免左右 Mesial 互吃 ---
        K_edge = idx.shape[-1]
        idx_expanded = idx.unsqueeze(1).expand(-1, self.out_c, -1, -1)
        neighbor_heatmap = torch.gather(heatmap_7.unsqueeze(-1).expand(-1, -1, -1, K_edge), 2, idx_expanded)
        local_max = heatmap_7 >= neighbor_heatmap.max(dim=-1)[0]

        max_class_heatmap = heatmap_7.max(dim=1, keepdim=True)[0]
        class_margin = heatmap_7 - max_class_heatmap
        cross_class_weight = torch.exp(class_margin / 0.2)

        peak_heatmap_7 = heatmap_7 * local_max.float() * cross_class_weight  # (B, 7, N)

        # =========================================================================
        # ★ 关键修改 2：通道降维融合 (7类 -> 6类)
        # 将通道 0 和 1 竞争合并，组成统一的 Mesial 通道参与后续 TopK 选举
        # =========================================================================
        # 1. 取 0 和 1 通道中的最大响应值和对应通道索引 (是左侧响应高还是右侧响应高？)
        mesial_max_val, mesial_max_idx = torch.max(peak_heatmap_7[:, 0:2, :], dim=1, keepdim=True)  # (B, 1, N)

        # 2. 融合 peak_heatmap：拼接 [Mesial_合并, 其它5类]
        peak_heatmap = torch.cat([mesial_max_val, peak_heatmap_7[:, 2:, :]], dim=1)  # (B, 6, N)

        # 3. 融合 offset_raw_7：依据最高响应的索引，抽取对应的偏移量
        idx_offset = mesial_max_idx.unsqueeze(2).expand(-1, -1, 3, -1)  # (B, 1, 3, N)
        offset_mesial = torch.gather(offset_raw_7[:, 0:2, :, :], dim=1, index=idx_offset)
        offset_raw = torch.cat([offset_mesial, offset_raw_7[:, 2:, :, :]], dim=1)  # (B, 6, 3, N)

        # 4. 融合基础 heatmap (供后续取权重时使用)
        heatmap_mesial = torch.gather(heatmap_7[:, 0:2, :], dim=1, index=mesial_max_idx)
        heatmap = torch.cat([heatmap_mesial, heatmap_7[:, 2:, :]], dim=1)  # (B, 6, N)

        # 5. 融合 cls_scores
        cls_scores_7_t = cls_scores_7.permute(0, 2, 1)  # (B, 7, N)
        cls_mesial = torch.gather(cls_scores_7_t[:, 0:2, :], dim=1, index=mesial_max_idx)
        cls_scores_merged = torch.cat([cls_mesial, cls_scores_7_t[:, 2:, :]], dim=1).permute(0, 2, 1)  # (B, N, 6)
        # =========================================================================

        # 选取每类 (现在是 6 类) 得分前 TopK 的点
        # 这里选取通道0时，会自动从合并后的库中抓取前50个点（不论它原本是属于左Mesial还是右Mesial）
        scores, topk_indices = torch.topk(peak_heatmap, k=self.topk, dim=-1)  # (B, 6, TopK)
        flat_topk_idx = topk_indices.view(batch_size, -1)  # (B, 6*TopK)
        num_candidates = flat_topk_idx.shape[1]

        # --- 后续所有代码完全使用 6 通道的融合特征 ---
        offset_permuted = offset_raw.detach().permute(0, 1, 3, 2).contiguous()  # (B, 6, N, 3)

        shifted_pos_all = raw_pos.unsqueeze(1) + offset_permuted
        topk_idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        select_shifted_pos = torch.gather(shifted_pos_all, 2, topk_idx_expanded)

        dist_per_class = torch.cdist(select_shifted_pos, shifted_pos_all)
        dist_to_all = dist_per_class.view(batch_size, -1, num_points)

        knn_dist_search, knn_idx_search = dist_to_all.topk(self.k_reg, dim=-1, largest=False)

        knn_dist_search = knn_dist_search.view(batch_size, self.classnums, self.topk, self.k_reg)
        knn_idx_search_c = knn_idx_search.view(batch_size, self.classnums, self.topk * self.k_reg)

        knn_idx_flat_search = knn_idx_search.view(batch_size, -1)
        knn_pos_search = torch.gather(raw_pos, 1, knn_idx_flat_search.unsqueeze(-1).expand(-1, -1, 3))
        knn_pos_search = knn_pos_search.view(batch_size, self.classnums, self.topk, self.k_reg, 3)

        knn_idx_search_c_exp = knn_idx_search_c.unsqueeze(-1).expand(-1, -1, -1, 3)
        knn_offset_search = torch.gather(offset_permuted, 2, knn_idx_search_c_exp)
        knn_offset_search = knn_offset_search.view(batch_size, self.classnums, self.topk, self.k_reg, 3)

        knn_pred_centers = knn_pos_search + knn_offset_search

        knn_heatmap_search = torch.gather(heatmap.detach(), 2, knn_idx_search_c)
        knn_heatmap_search = knn_heatmap_search.view(batch_size, self.classnums, self.topk, self.k_reg)

        dist_threshold = 1.2
        dist_mask = (knn_dist_search < dist_threshold).float()

        weights = knn_heatmap_search * dist_mask
        weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-6
        weights_norm = weights / weights_sum

        select_center_pos_c = torch.sum(weights_norm.unsqueeze(-1) * knn_pred_centers, dim=-2)
        select_center_pos = select_center_pos_c.view(batch_size, -1, 3)

        dist = torch.cdist(select_center_pos, raw_pos)
        _, knn_idx_reg = dist.topk(self.k_reg, dim=-1, largest=False)
        knn_idx_reg_flat = knn_idx_reg.view(batch_size, -1)

        knn_pos = torch.gather(raw_pos, 1, knn_idx_reg_flat.unsqueeze(-1).expand(-1, -1, 3))
        knn_pos = knn_pos.view(batch_size, num_candidates, self.k_reg, 3)

        final_features_t = final_features.permute(0, 2, 1)
        knn_feat = torch.gather(final_features_t, 1, knn_idx_reg_flat.unsqueeze(-1).expand(-1, -1, self.dim * 16))
        knn_feat = knn_feat.view(batch_size, num_candidates, self.k_reg, self.dim * 16)

        # 注意这里替换为 cls_scores_merged
        knn_cls = torch.gather(cls_scores_merged.detach(), 1,
                               knn_idx_reg_flat.unsqueeze(-1).expand(-1, -1, self.classnums))
        knn_cls = knn_cls.view(batch_size, num_candidates, self.k_reg, self.classnums)

        rel_pos = knn_pos - select_center_pos.unsqueeze(2)
        dist_pos = torch.norm(rel_pos, dim=-1, keepdim=True)
        rel_pos_with_dist = torch.cat([rel_pos, dist_pos], dim=-1)

        rel_pos_emb = self.pos_embed(rel_pos_with_dist.permute(0, 3, 1, 2).contiguous())
        knn_feat_emb = self.mapliner(knn_feat.permute(0, 3, 1, 2).contiguous())

        cls_idx = torch.arange(self.classnums, device=x.device).view(1, self.classnums, 1).expand(batch_size,
                                                                                                  self.classnums,
                                                                                                  self.topk)
        target_cls_emb = self.class_emb(cls_idx.reshape(batch_size, -1))
        target_cls_emb = target_cls_emb.unsqueeze(-1).expand(-1, -1, -1, self.k_reg).permute(0, 2, 1, 3).contiguous()

        grouped_features = torch.cat([knn_feat_emb, rel_pos_emb, target_cls_emb], dim=1)

        reg_hidden = self.local_refiner(grouped_features)
        reg_hidden = reg_hidden.view(batch_size, self.classnums, self.topk, -1)

        cls = self.clsLiner(inverse_sigmoid(scores.unsqueeze(dim=-1)))
        reg_conf = self.predconf(reg_hidden)
        pred_delta = self.predland(reg_hidden)

        select_center_pos_reshaped = select_center_pos.view(batch_size, self.classnums, self.topk, 3)
        final_pred_land = select_center_pos_reshaped + pred_delta

        # 返回第一阶段产生的 7 通道密集预测结果用于 Loss，返回 6 类的地标预测给后处理
        heatmap_return_7 = heatmap_7.permute(0, 2, 1).contiguous()

        return out_seg, heatmap_return_7, offset_return_7, cls_scores_7, final_pred_land, reg_conf, pred_delta, select_center_pos_reshaped