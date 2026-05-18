import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        # 如果未提供idx，则基于当前特征计算KNN（动态图特性）
        # 如果是原始输入，仅使用前3维计算；如果是深层特征，使用全部维度计算
        if x.size(1) == 3:
            idx = knn(x, k=k)
        else:
            idx = knn(x, k=k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idxx = idx + idx_base
    idxx = idxx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idxx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # 拼接 (neighbor - center, center)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature, idx


class EdgeConvBlock(nn.Module):
    """带 GroupNorm 和残差连接的 EdgeConv 模块"""

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
        x = x.max(dim=-1, keepdim=False)[0]  # 局部聚合
        x = x + self.shortcut(identity)  # 残差连接
        return x, idx


class ChannelAttention(nn.Module):
    """轻量级通道注意力机制 (类似SE Block)，增强关键特征表达"""

    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ToothLandmark(nn.Module):
    def __init__(self, classnums=6, k=20, emb_dims=1024):
        super(ToothLandmark, self).__init__()
        self.k = k
        self.dim = 64
        groups = 8

        # 1. 局部特征提取层 (EdgeConv)
        # 维持原有通道数不减少：3 -> 64 -> 128 -> 256 -> 256
        self.edge_conv1 = EdgeConvBlock(3, self.dim, k=k, groups=groups)
        self.edge_conv2 = EdgeConvBlock(self.dim, self.dim * 2, k=k, groups=groups)
        self.edge_conv3 = EdgeConvBlock(self.dim * 2, self.dim * 4, k=k, groups=groups)
        self.edge_conv4 = EdgeConvBlock(self.dim * 4, self.dim * 4, k=k, groups=groups)

        # 2. 局部特征聚合与全局映射
        # local_features 维度: 64 + 128 + 256 + 256 = 704
        self.local_dim = self.dim * 11

        # 将局部特征映射到真正的 emb_dims (1024)，修复原代码中仅映射到256的问题
        self.agg_conv = nn.Sequential(
            nn.Conv1d(self.local_dim, emb_dims, kernel_size=1, bias=False),
            nn.GroupNorm(groups, emb_dims),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. 最终特征融合与解码器头部
        # 融合维度 = Global Max(1024) + Global Avg(1024) + Local(704) + 原始坐标(3) = 2755
        self.fusion_dim = emb_dims * 2 + self.local_dim + 3

        # 将融合后的超高维特征降维压缩到 1024 维 (原代码 self.dim*16 即 1024)
        self.conv7 = nn.Sequential(
            nn.Conv1d(self.fusion_dim, self.dim * 16, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 16),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 加入通道注意力，让网络更加关注对定位敏感的特征
        self.attention = ChannelAttention(self.dim * 16)

        # Heatmap 分支保持原通道数: 1024 -> 512 -> 256 -> classnums
        self.heatmap_head1 = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, (classnums-1), kernel_size=1)
        )

        # Offset 分支保持原通道数: 1024 -> 512 -> 256 -> classnums*3
        self.offset_head1 = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, (classnums-1) * 3, kernel_size=1)
        )
        self.class_t = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, classnums, kernel_size=1))
    def forward(self, x):
        # x 预期输入 shape: (B, 3, N)
        x = x.permute(0, 2, 1) #
        batch_size = x.size(0)
        num_points = x.size(2)
        raw_pts = x  # 保存原始坐标用于 Skip-Connection

        # --- 多尺度特征提取 (恢复部分动态图特性) ---
        # 浅层提取几何特征（依赖空间KNN）
        x1, idx1 = self.edge_conv1(x, idx=None)
        x2, idx2 = self.edge_conv2(x1, idx=idx1)

        # 深层提取语义特征（将 idx 置为 None，触发动态图根据高维特征重新寻找近邻）
        x3, idx3 = self.edge_conv3(x2, idx=None)
        x4, _ = self.edge_conv4(x3, idx=idx3)

        # 拼接多尺度局部特征: (B, 704, N)
        local_features = torch.cat((x1, x2, x3, x4), dim=1)

        # --- 提取双重全局特征 (Dual Global Context) ---
        high_level_feat = self.agg_conv(local_features)  # (B, 1024, N)

        # 1. Max Pooling (显著特征)
        global_max = high_level_feat.max(dim=-1, keepdim=True)[0]  # (B, 1024, 1)
        # 2. Avg Pooling (分布特征)
        global_avg = high_level_feat.mean(dim=-1, keepdim=True) # (B, 1024, 1)

        global_feature = torch.cat([global_max, global_avg], dim=1)  # (B, 2048, 1)
        global_feature = global_feature.repeat(1, 1, num_points)  # (B, 2048, N)

        # --- 最终特征融合 (Global + Local + Raw Coordinates) ---
        # 融合维度: 2048 + 704 + 3 = 2755
        final_features = torch.cat((global_feature, local_features, raw_pts), dim=1)

        # 降维回 1024 并应用注意力机制
        final_features = self.conv7(final_features)  # (B, 1024, N)
        final_features = self.attention(final_features)

        # --- 输出预测 ---
        heatmap = torch.sigmoid(self.heatmap_head1(final_features)).permute(0, 2, 1)  # (B, N, C)
        offset = self.offset_head1(final_features).permute(0, 2, 1)
        offset = offset.reshape(batch_size, num_points, -1, 3)  # (B, N, C, 3)
        cls = torch.sigmoid(self.class_t(final_features).permute(0, 2, 1))  # (B, N, C)
        return heatmap, offset, cls