import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from functools import partial
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment  # Assuming you use this in loss


# --- 基础工具函数 ---

def knn(x, k):
    # x: (B, 3, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, K)
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


# --- 主模型 ---

class ToothLandmark(nn.Module):
    def __init__(self, classnums=6, k=20, topk=50):
        super(ToothLandmark, self).__init__()
        self.k = k
        self.topk = topk
        self.dim = 64
        self.classnums = classnums
        groups = 8

        # 1. 骨干网络 (EdgeConv)
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

        # 3. 原始热图分支
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
        self.k_reg = 16

        self.pos_embed = nn.Sequential(
            nn.Conv2d(3, self.dim, kernel_size=1),
            nn.GroupNorm(8, self.dim),
            nn.LeakyReLU(0.2)
        )
        self.score_embed = nn.Sequential(
            nn.Conv2d(self.classnums * 2, self.dim, kernel_size=1),
            nn.GroupNorm(8, self.dim),
            nn.LeakyReLU(0.2)
        )

        # --- 新增: 目标类别 Embedding ---
        # 显式告诉 MLP 当前候选点是在预测哪一个类别，解决类别混淆问题
        self.class_emb = nn.Embedding(self.classnums, self.dim)

        # 局部特征聚合MLP
        # 输入: 邻居特征 dim*16 + 相对坐标 dim + 概率先验 dim + 类别 Embedding dim
        reg_in_dim = self.dim * 16 + self.dim + self.dim + self.dim

        self.local_mlp = nn.Sequential(
            nn.Conv2d(reg_in_dim, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim * 8, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 候选点精细化头
        self.reg_refine1 = nn.Sequential(
            nn.Linear(self.dim * 8, self.dim * 4),
            nn.LayerNorm(self.dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim * 4, self.dim * 4),
            nn.LeakyReLU(0.2))

        self.predconf = nn.Linear(self.dim * 4, 1)
        self.predland = nn.Linear(self.dim * 4, 3)

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

        # --- 原始分支输出 ---
        heatmap = torch.sigmoid(self.heatmap_head(final_features))  # (B, C, N)
        offset = self.offset_head(final_features).reshape(batch_size, self.classnums, 3, num_points).permute(0, 3, 1, 2)
        cls_scores = torch.sigmoid(self.class_t(final_features)).permute(0, 2, 1)  # (B, N, C)

        # --- 直接坐标回归分支 ---
        B, C, N = heatmap.shape
        K_edge = idx.shape[-1]

        # 1. 局部极大值筛选
        idx_expanded = idx.unsqueeze(1).expand(-1, C, -1, -1)
        neighbor_heatmap = torch.gather(heatmap.unsqueeze(-1).expand(-1, -1, -1, K_edge), 2, idx_expanded)
        local_max = heatmap >= neighbor_heatmap.max(dim=-1)[0]
        peak_heatmap = heatmap * local_max.float()

        # 2. 选取每类 Peak 中得分前 TopK 的点
        scores, topk_indices = torch.topk(peak_heatmap, k=self.topk, dim=-1)  # (B, C, TopK)
        flat_topk_idx = topk_indices.view(batch_size, -1)  # (B, C*TopK)
        num_candidates = flat_topk_idx.shape[1]

        # 提取候选点中心坐标: (B, C*TopK, 3)
        select_pos = torch.gather(raw_pos, 1, flat_topk_idx.unsqueeze(-1).expand(-1, -1, 3))

        # --- 新增: 生成目标类别 Embedding ---
        # 对应每个候选点，生成它所属类别的 Embedding
        cls_idx = torch.arange(self.classnums, device=x.device).view(1, self.classnums, 1).expand(batch_size,
                                                                                                  self.classnums,
                                                                                                  self.topk)
        cls_idx_flat = cls_idx.reshape(batch_size, -1)  # (B, C*TopK)
        target_cls_emb = self.class_emb(cls_idx_flat)  # (B, C*TopK, dim)
        # 扩展 K_reg 维度以便后续拼接
        target_cls_emb = target_cls_emb.unsqueeze(2).expand(-1, -1, self.k_reg, -1)  # (B, C*TopK, K_reg, dim)

        # 3. 提取 K_reg 近邻特征
        dist = torch.cdist(select_pos, raw_pos)
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

        # 坐标和得分的 Embed
        rel_pos = knn_pos - select_pos.unsqueeze(2)
        rel_pos = self.pos_embed(rel_pos.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        knn_cls_embed = self.score_embed(
            torch.cat([knn_heatmap, knn_cls], dim=-1).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        # --- 修改: 拼接时加入 target_cls_emb ---
        grouped_features = torch.cat([knn_feat, rel_pos, knn_cls_embed, target_cls_emb], dim=-1)
        grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()

        # 4. 局部 MLP 与池化
        agg_feat = self.local_mlp(grouped_features)
        agg_feat = agg_feat.max(dim=-1)[0]

        reg_hidden = agg_feat.permute(0, 2, 1).contiguous()
        reg_hidden = reg_hidden.view(batch_size, self.classnums, self.topk, -1)

        # 5. 回归与置信度预测
        reg_out = self.reg_refine1(reg_hidden)

        reg_conf = self.predconf(reg_out)
        pred_delta = self.predland(reg_out)

        select_pos_reshaped = select_pos.view(batch_size, self.classnums, self.topk, 3)
        final_pred_land = select_pos_reshaped + pred_delta

        return heatmap_t, offset, cls_scores, final_pred_land, reg_conf, pred_delta, select_pos_reshaped