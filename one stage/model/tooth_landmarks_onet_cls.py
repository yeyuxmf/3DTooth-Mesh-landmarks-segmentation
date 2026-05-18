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
        # 仅使用坐标(前3维)计算KNN
        idx = knn(x[:, :3, :], k=k)

    device = x.device  # 动态获取设备，避免硬编码cuda
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
        # 如果输入输出维度不一致，用1x1卷积做shortcut映射
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


class ToothLandmark(nn.Module):
    def __init__(self, classnums=6, k=20, emb_dims=1024):
        super(ToothLandmark, self).__init__()
        self.k = k
        self.dim = 64
        groups = 8  # GN的组数

        # 1. 局部特征提取层 (EdgeConv)
        self.edge_conv1 = EdgeConvBlock(3, self.dim, k=k, groups=groups)
        self.edge_conv2 = EdgeConvBlock(self.dim, self.dim* 2, k=k, groups=groups)
        self.edge_conv3 = EdgeConvBlock(self.dim * 2, self.dim * 4, k=k, groups=groups)
        self.edge_conv4 = EdgeConvBlock(self.dim * 4, self.dim * 4, k=k, groups=groups)

        # 2. 特征融合层
        # 融合维度: 64 + 64 + 128 + 128 = 384
        fusion_dim = self.dim * 11
        self.agg_conv = nn.Sequential(
            nn.Conv1d(fusion_dim, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. 解码器头部 (Heatmap & Offset)
        # 拼接全局特征后的总维度: emb_dims + fusion_dim = 1024 + 384 = 1408
        self.head_dim = self.dim * 4 + fusion_dim
        self.conv7 = nn.Sequential(nn.Conv1d(self.head_dim,  self.dim * 16, kernel_size=1, bias=False),
                                   nn.GroupNorm(groups, self.dim * 16),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.heatmap_head = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, classnums-1, kernel_size=1))

        self.offset_head = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, (classnums-1)*3, kernel_size=1))


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
        # x shape: (B, 3, N)
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)
        num_points = x.size(2)

        # 多尺度特征提取
        x1, idx = self.edge_conv1(x, idx=None)  # (B, 64, N)
        x2, idx = self.edge_conv2(x1, idx=idx)  # (B, 64, N)
        x3, idx = self.edge_conv3(x2, idx=idx)  # (B, 128, N)
        x4, idx = self.edge_conv4(x3, idx=idx)  # (B, 128, N)

        # 拼接多尺度局部特征
        local_features = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 384, N)

        # 提取全局特征
        combined = self.agg_conv(local_features)  # (B, 1024, N)
        global_feature = combined.max(dim=-1, keepdim=True)[0]  # (B, 1024, 1)
        global_feature = global_feature.repeat(1, 1, num_points)  # (B, 1024, N)

        # 最终特征融合 (Global + Local)
        final_features = torch.cat((global_feature, local_features), dim=1)  # (B, 1408, N)
        final_features = self.conv7(final_features)

        # 输出预测
        heatmap = torch.sigmoid(self.heatmap_head(final_features)).permute(0, 2, 1)  # (B, N, C)
        offset = self.offset_head(final_features).permute(0, 2, 1)
        offset = offset.reshape(batch_size, num_points, -1, 3)  # (B, N, C, 3)

        cls = torch.sigmoid(self.class_t(final_features).permute(0, 2, 1))  # (B, N, C)

        return heatmap, offset, cls