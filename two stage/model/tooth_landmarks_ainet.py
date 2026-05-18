import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block

# 保持你的自定义引用不变
from model.pos_embed import PositionalEncoding
from model.jitblock import JiTBlock, RMSNorm
from model.transformer import Transformer
from model.loss import matchLandmarkLoss, fixedLandmarkLoss
from config import config


# ==========================================
# 1. 核心模块增强: EdgeConv (捕捉局部边缘几何)
# ==========================================
def knn_point(x, k):
    """
    针对点云自身的 KNN (用于 Backbone)
    x: (B, C, N)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


class EdgeConvBlock(nn.Module):
    """
    DGCNN 风格的边缘卷积，显著增强对牙齿边缘和尖端的感知能力
    """

    def __init__(self, in_channels, out_channels, k=16):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        # x: (B, N, C) -> (B, C, N)
        x = x.transpose(1, 2)
        batch_size = x.size(0)
        num_points = x.size(2)

        idx = knn_point(x, self.k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2, 1).contiguous()  # (B, N, C)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, -1)
        x = x.view(batch_size, num_points, 1, -1).repeat(1, 1, self.k, 1)

        # Concat (x_neighbor - x_center, x_center)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        x = self.conv(feature)  # (B, C_out, N, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C_out, N)
        return x.transpose(1, 2)  # (B, N, C_out)


# ==========================================
# 2. 相对位置编码器 (升级版: 包含距离信息)
# ==========================================
class RelativePosEncoder(nn.Module):
    def __init__(self, in_dim=4, out_dim=64):
        # 注意: in_dim 改为 4 (dx, dy, dz, distance)
        super().__init__()
        self.mlps = nn.Sequential(
            nn.Linear(in_dim, out_dim // 2),
            nn.LayerNorm(out_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(out_dim // 2, out_dim)
        )

    def forward(self, x):
        return self.mlps(x)


# ==========================================
# 3. Stage 2 核心: Cross Attention Refinement
# ==========================================
class CrossAttnRefineBlock(nn.Module):
    """
    替代 MaxPool。使用 Attention 机制，让 Center(Query) 动态决定
    哪些 Neighbor(Key/Value) 对最终位置修正更重要。
    """

    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)  # Query: 粗预测点特征
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)  # Key: 邻域点特征
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)  # Value: 邻域点特征

        self.proj = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim)

    def forward(self, center_feat, neighbor_feat):
        # center_feat: (Batch_Total, 1, C)
        # neighbor_feat: (Batch_Total, K, C)
        B, K, C = neighbor_feat.shape

        # Residual connection
        residual = center_feat

        q = self.q_proj(center_feat).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(neighbor_feat).reshape(B, K, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(neighbor_feat).reshape(B, K, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Attention: (B, Heads, 1, K)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Aggregate: (B, Heads, 1, Head_Dim) -> (B, 1, C)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)

        # Add & Norm
        x = self.norm(x + residual)
        return x


# ==========================================
# 4. 功能函数 (KNN Cross & Feature Gather)
# ==========================================
def knn_cross(init_coords, input_pos, k):
    """
    Stage 2 使用的 KNN: 找 init_coords 周围的 input_pos
    """
    dist_init = torch.sum(init_coords ** 2, dim=-1, keepdim=True)  # (B, M, 1)
    dist_input = torch.sum(input_pos ** 2, dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, N)
    inner = torch.matmul(init_coords, input_pos.transpose(1, 2))  # (B, M, N)
    pairwise_distance = dist_init + dist_input - 2 * inner
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (B, M, k)
    return idx


def get_local_feature_with_geometry(init_coords, input_pos, input_feat, idx, pos_encoder):
    """
    增强版特征提取：加入欧氏距离作为显式特征
    """
    TN, M, K = idx.shape
    C = input_feat.shape[-1]
    device = input_feat.device

    batch_indices = torch.arange(TN, device=device).view(TN, 1, 1).expand(TN, M, K)

    # Gather
    neighbor_feat = input_feat[batch_indices, idx, :]  # (TN, M, K, C)
    neighbor_xyz = input_pos[batch_indices, idx, :]  # (TN, M, K, 3)

    # 1. 相对坐标
    delta_xyz = neighbor_xyz - init_coords.unsqueeze(2)  # (TN, M, K, 3)

    # 2. 欧氏距离 (Distance Awareness) - 这是一个非常强的几何线索
    dist = torch.norm(delta_xyz, dim=-1, keepdim=True)  # (TN, M, K, 1)

    # 3. 编码 (输入维度 3+1=4)
    geo_info = torch.cat([delta_xyz, dist], dim=-1)  # (TN, M, K, 4)
    pos_feat = pos_encoder(geo_info)

    # 4. 融合
    combined_feat = neighbor_feat + pos_feat

    return combined_feat


# ==========================================
# 5. 预测头
# ==========================================
class LandmarkPredictor(nn.Module):
    def __init__(self, input_channels, key_nums=5):
        super(LandmarkPredictor, self).__init__()
        self.num_heads = key_nums
        in_features = input_channels
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.LeakyReLU(),
                nn.Linear(in_features // 2, 7)
            ) for _ in range(key_nums)
        ])

    def forward(self, x):
        results = []
        for i in range(self.num_heads):
            results.append(self.heads[i](x))
        return torch.stack(results, dim=-2)


class oLandmarkPredictor(nn.Module):
    def __init__(self, input_channels, key_nums=5):
        super(oLandmarkPredictor, self).__init__()
        # Stage 2 回归 Offset
        in_features = input_channels
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.LeakyReLU(),
                nn.Linear(in_features // 2, 7)
            ) for _ in range(2)
        ])

    def forward(self, x, x1):
        results = []
        results.append(self.heads[0](x))
        results.append(self.heads[1](x1))
        return torch.cat(results, dim=-2)

class GeometricResidual(nn.Module):
    def __init__(self, embed_dim):
        super(GeometricResidual, self).__init__()
        # 捕捉点云的绝对位置和相对重心的位置
        # 输入 3 维坐标，输出 embed_dim 维特征
        self.geo_mlp = nn.Sequential(
            nn.Linear(3, embed_dim // 4),
            nn.BatchNorm1d(embed_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        self.alpha = nn.Parameter(torch.full((1,), 0.1)) # 初始给一个很小的权重，让模型自己学习注入多少

    def forward(self, x):
        # x: (TB*TN, N, 3)
        B, N, C = x.shape
        x_flat = x.view(-1, C)
        feat = self.geo_mlp(x_flat)
        feat = feat.view(B, N, -1)
        return self.alpha * feat
# ==========================================
# 6. 主模型 ToothLandmark (完整结构)
# ==========================================
class ToothLandmark(nn.Module):
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=2):
        super(ToothLandmark, self).__init__()

        self.fxid_tnums = config.fxid_tnums
        self.match_tnums = config.match_tnums
        self.fm_tnums = config.fxid_tnums + config.match_tnums
        self.knn_nums = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Part 1: Point Cloud Encoder ---
        self.linear1 = nn.Linear(3, embed_dim)

        # [NEW] EdgeConv 模块: 在进入 Transformer 前提取局部几何特征
        self.geo_res = GeometricResidual(embed_dim)

        self.linear2 = nn.Linear(3, embed_dim)
        norm_layer = partial(RMSNorm, eps=1e-6)

        # Backbone
        self.Tsencoders = nn.ModuleList([
            Block(dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.Tcurencoders = nn.ModuleList([
            Block(dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.linearB = nn.Linear(embed_dim, embed_dim)

        self.Tcencoders = nn.ModuleList([
            Transformer(dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, qkv_bias=True,
                        norm_layer=norm_layer)
            for i in range(depth)])

        self.encoder_embed = nn.Parameter(torch.zeros(1, config.tooth_nums, embed_dim), requires_grad=False)

        self.trans_ = nn.ModuleList([
            JiTBlock(hidden_size=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, attn_drop=0.0,
                     proj_drop=0.0)
            for i in range(depth)])

        # --- Stage 1 Heads ---
        self.linearMap = nn.Linear(embed_dim * 2, embed_dim)
        self.linearMap1 = nn.Linear(embed_dim, embed_dim * 2)
        self.linearh = nn.Linear(1, embed_dim)
        self.heat_out = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(embed_dim, 1))

        self.predlandmarks = LandmarkPredictor(input_channels=embed_dim * 2, key_nums=self.fm_tnums)

        # --- Stage 2 Modules (Refinement) ---

        # [NEW] 相对位置编码器输入改为 4 (dx, dy, dz, dist)
        self.rel_pos_encoder = RelativePosEncoder(in_dim=4, out_dim=embed_dim)

        # [NEW] Query 编码器 (将 Stage 1 的粗糙坐标编码为 Query 特征)
        self.query_pos_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 降维映射
        self.lineargh = nn.Linear(embed_dim, embed_dim)

        # [NEW] Attention Refinement Block (替代原有的 Transformer Block 堆叠 + MaxPool)
        # 这里使用 2 层 Cross Attention 进行强力精修
        self.refine_layers = nn.ModuleList([
            CrossAttnRefineBlock(dim=embed_dim, num_heads=4)
            for _ in range(2)
        ])

        self.lineargdh = nn.Linear(embed_dim, embed_dim * 2)
        self.outLandmarks = oLandmarkPredictor(input_channels=embed_dim * 2, key_nums=self.fm_tnums)

        # Losses
        self.match_loss = matchLandmarkLoss()
        self.fixed_loss = fixedLandmarkLoss()

        self._init_weights(self.predlandmarks)
        self._init_weights(self.outLandmarks)
        self._init_weights(self.rel_pos_encoder)
        self._init_weights(self.query_pos_embed)
        self._init_weights(self.refine_layers)
        self._init_weights(self.geo_res)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def initialize_weights(self):
        pos_embed1 = PositionalEncoding(self.encoder_embed.shape[1], self.encoder_embed.shape[2], self.device)
        self.encoder_embed.data.copy_(pos_embed1.float().unsqueeze(0))

    def forward(self, p_points, trans_mask):
        TB, TN, N, C = p_points.shape
        input_pos = p_points[..., :3].clone()
        trans_mask = trans_mask > 0

        # --- Pre-processing ---
        cp = torch.mean(p_points[..., :3], dim=-2)
        if self.training:
            noise = (torch.randn(TB * 1, TN, 3).cuda().float() * 1.0)
            cp = cp + noise
        # 1. 获取中心化后的坐标
        p_points_centered = p_points[..., :3] - cp.unsqueeze(dim=-2)
        # p_points_centered 形状为 (TB, TN, N, 3)

        # 2. 准备 flatten 数据给 Transformer
        points_xyz = p_points_centered.reshape(TB * TN, N, -1)

        # 3. 线性映射 + 几何残差注入
        # x_base 是原始的语义 Embedding
        x_base = self.linear1(points_xyz)
        points_feat = x_base + self.geo_res(points_xyz)

        cpoints = self.linear2(cp)

        # Transformer Layers
        teeths = []
        for blk in self.Tsencoders:
            points_feat = blk(points_feat)
            teeths.append(points_feat)

        for blk in self.Tcencoders:
            cpoints = blk(cpoints, trans_mask)

        x = teeths[-1]  # (TB*TN, N, embed_dim)
        x = self.linearB(x)

        condtion = cpoints.reshape(TB * TN, -1).unsqueeze(dim=1)

        xxin = []
        for blk in self.trans_:
            x = blk(x, condtion)
            xxin.append(x)

        # --- Stage 1 Output ---
        heatMap = self.heat_out(xxin[-1])
        heat_out = heatMap.reshape(TB, TN, config.sam_points, 1).sigmoid()

        linearh = self.linearh(heatMap)
        x_combined = torch.cat([linearh, torch.cat(xxin[-1:], dim=-1)], dim=-1)

        # 保存点云特征供 Stage 2 使用
        point_features = self.linearMap(x_combined)

        x_global = torch.max(point_features, dim=1)[0]
        x_global = self.linearMap1(x_global)

        predlandmarks = self.predlandmarks(x_global).reshape(TB, TN, self.fm_tnums, -1)

        predlandmarks_abs = predlandmarks.clone()
        predlandmarks_abs[..., :3] = predlandmarks[..., :3] + cp.unsqueeze(dim=-2)

        fixedlandmarks = predlandmarks_abs[..., :config.fxid_tnums, :].reshape(TB, TN, -1)
        matchLandmarks = predlandmarks_abs[..., config.fxid_tnums:, :].reshape(TB, TN, -1)

        # =========================================================
        # Stage 2: Refinement (重构的核心部分)
        # =========================================================

        # 1. 准备数据
        # init_coords: 粗预测中心 (Batch*TN, Landmarks, 3)
        init_coords = predlandmarks_abs[..., :3].reshape(TB * TN, self.fm_tnums, 3).detach()
        input_pos_flat = input_pos.reshape(TB * TN, config.sam_points, 3)

        # 2. KNN Search
        idx = knn_cross(init_coords, input_pos_flat, k=self.knn_nums)

        # 3. 几何感知特征提取 (含 Distance)
        # neighbor_feat: (TB*TN, Landmarks, K, embed_dim)
        neighbor_feat = get_local_feature_with_geometry(init_coords, input_pos_flat, point_features, idx,
                                                        self.rel_pos_encoder)
        neighbor_feat = self.lineargh(neighbor_feat)

        # 4. 构造 Query
        # 使用 Stage 1 预测的坐标生成 Query 特征
        # (TB*TN, Landmarks, 3) -> (TB*TN, Landmarks, 1, embed_dim)
        query_feat = self.query_pos_embed(init_coords).unsqueeze(2)

        # 5. Reshape for Attention
        # 我们把 (Batch * Teeth * Landmarks) 视为 Attention 的 Batch 维度
        # 这样每组 (Query=1, Keys=K) 独立做 Attention
        B_large = TB * TN * self.fm_tnums

        # (B_large, 1, C)
        query_flat = query_feat.view(B_large, 1, -1)
        # (B_large, K, C)
        neighbor_flat = neighbor_feat.view(B_large, self.knn_nums, -1)

        # 6. Cross Attention Refinement
        refined_feat = query_flat
        for blk in self.refine_layers:
            # Query 关注 Neighbor，聚合信息
            refined_feat = blk(refined_feat, neighbor_flat)

        # (B_large, 1, C) -> (B_large, C)
        refined_feat = refined_feat.squeeze(1)

        # 7. Reshape Back & Predict
        refined_feat = refined_feat.view(TB * TN, self.fm_tnums, -1)
        refined_feat = self.lineargdh(refined_feat)  # 升维

        fland_fea = refined_feat[:, :config.fxid_tnums, :]
        mland_fea = refined_feat[:, config.fxid_tnums:, :]

        # 预测 Offset
        land_offsets = self.outLandmarks(fland_fea, mland_fea)

        # 叠加 Offset 到粗坐标
        land_offsets[..., :3] = land_offsets[..., :3] + init_coords

        final_coords = land_offsets.reshape(TB, TN, self.fm_tnums, -1)
        flandmarks_refine = final_coords[..., :config.fxid_tnums, :].reshape(TB, TN, -1)
        mLandmarks_refine = final_coords[..., config.fxid_tnums:, :].reshape(TB, TN, -1)

        return fixedlandmarks, matchLandmarks, heat_out, flandmarks_refine, mLandmarks_refine