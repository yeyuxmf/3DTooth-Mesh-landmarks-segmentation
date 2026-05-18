import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block

# 假设这些是你自定义的模块，保持引用不变
from model.pos_embed import PositionalEncoding
from model.jitblock import JiTBlock, RMSNorm
from model.transformer import Transformer
from model.loss import matchLandmarkLoss, fixedLandmarkLoss
from config import config


# ==========================================
# 1. 相对位置编码器 (New)
# ==========================================
class RelativePosEncoder(nn.Module):
    """
    将 3D 相对坐标 (dx, dy, dz) 映射到高维特征空间
    """

    def __init__(self, in_dim=3, out_dim=64):
        super().__init__()
        self.mlps = nn.Sequential(
            nn.Linear(in_dim, out_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(out_dim // 2, out_dim)
        )

    def forward(self, x):
        return self.mlps(x)


# ==========================================
# 2. KNN 搜索 (优化版)
# ==========================================
def knn_cross(init_coords, input_pos, k):
    """
    init_coords: (B, M, 3) - 中心点
    input_pos:   (B, N, 3) - 源点云
    """
    # 距离计算: ||a - b||^2 = a^2 + b^2 - 2ab
    dist_init = torch.sum(init_coords ** 2, dim=-1, keepdim=True)  # (B, M, 1)
    dist_input = torch.sum(input_pos ** 2, dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, N)
    inner = torch.matmul(init_coords, input_pos.transpose(1, 2))  # (B, M, N)

    pairwise_distance = dist_init + dist_input - 2 * inner

    # 取最近的 k 个
    # largest=False 表示取最小距离
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (B, M, k)
    return idx


# ==========================================
# 3. 几何感知特征提取 (Core Fix)
# ==========================================
def get_local_feature_with_geometry(init_coords, input_pos, input_feat, idx, pos_encoder):
    """
    提取局部特征，并注入相对位置信息
    init_coords: (TN, M, 3) - Stage 1 预测中心
    input_pos:   (TN, N, 3) - 原始点云坐标
    input_feat:  (TN, N, C) - 原始点云特征
    idx:         (TN, M, K) - KNN 索引
    pos_encoder: nn.Module  - 相对位置编码器
    """
    TN, M, K = idx.shape
    C = input_feat.shape[-1]
    device = input_feat.device

    # 1. 构造 Batch 索引
    batch_indices = torch.arange(TN, device=device).view(TN, 1, 1).expand(TN, M, K)

    # 2. Gather 邻域特征 (TN, M, K, C)
    neighbor_feat = input_feat[batch_indices, idx, :]

    # 3. Gather 邻域坐标 (TN, M, K, 3)
    neighbor_xyz = input_pos[batch_indices, idx, :]

    # 4. 计算相对坐标 (Relative Coordinates)
    # 这一步至关重要：告诉网络每个邻域点相对于预测中心在哪里
    # init_coords: (TN, M, 1, 3)
    delta_xyz = neighbor_xyz - init_coords.unsqueeze(2)

    # 5. 编码相对位置 (TN, M, K, C)
    # 假设 pos_encoder 输出维度与 semantic feature 维度一致，如果不一致可以使用 concat
    delta_feat = pos_encoder(delta_xyz)

    # 6. 特征融合
    # 将“我在哪里(delta)”和“我是什么(neighbor_feat)”结合
    # 这里使用相加 (类似于 Transformer 的 Positional Embedding)，也可以 concat
    combined_feat = neighbor_feat + delta_feat

    # 7. 聚合特征
    # 现在 MaxPool 聚合的是包含位置信息的特征
    #weighted_feat = combined_feat.max(dim=-2)[0]  # (TN, M, C)

    return combined_feat


# ==========================================
# 4. 预测头 (保持不变)
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
                nn.Linear(in_features // 2, 7)  # 仅预测 xyz
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
        # Stage 2 用于回归 Offset
        in_features = input_channels
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.LeakyReLU(),
                nn.Linear(in_features // 2, 7)
            ) for _ in range(2)  # 0: fixed, 1: match
        ])

    def forward(self, x, x1):
        results = []
        results.append(self.heads[0](x))
        results.append(self.heads[1](x1))
        return torch.cat(results, dim=-2)


# ==========================================
# 5. 主模型 ToothLandmark
# ==========================================
class ToothLandmark(nn.Module):
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=2):
        super(ToothLandmark, self).__init__()

        self.fxid_tnums = config.fxid_tnums
        self.match_tnums = config.match_tnums
        self.fm_tnums = config.fxid_tnums + config.match_tnums
        self.knn_nums = 32

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Point Cloud Encoder ---
        self.linear1 = nn.Linear(3, embed_dim)
        self.linear2 = nn.Linear(3, embed_dim)

        norm_layer = partial(RMSNorm, eps=1e-6)

        # Backbone Blocks
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

        # Embeddings & Parameters
        self.encoder_embed = nn.Parameter(torch.zeros(1, config.tooth_nums, embed_dim), requires_grad=False)

        self.trans_ = nn.ModuleList([
            JiTBlock(hidden_size=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, attn_drop=0.0,
                     proj_drop=0.0)
            for i in range(depth)])

        # --- Stage 1 Heads ---
        # 这里的 embed_dim*4 是 linearMap 的输出维度
        self.s1_refinement = nn.Linear(embed_dim * 2, embed_dim)  # 专门给 S1 用
        self.s2_refinement = nn.Linear(embed_dim * 2, embed_dim)  # 专门给 S2 用

        self.linearMap1 = nn.Linear(embed_dim, embed_dim * 2)

        self.linearh = nn.Linear(1, embed_dim)
        self.heat_out = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(embed_dim, 1))

        self.predlandmarks = LandmarkPredictor(input_channels=embed_dim * 2,
                                               key_nums=self.fm_tnums)

        # --- Stage 2 Modules (关键修改部分) ---

        # 1. 相对位置编码器：输出维度要和 feature 维度对齐以便融合
        # 这里的 feature 是来自 linearMap 的输出 (embed_dim*4)
        self.rel_pos_encoder = RelativePosEncoder(in_dim=3, out_dim=embed_dim)

        # 2. 降维映射：将融合后的特征 (embed_dim*4) 降回 embed_dim 供 Transformer 使用
        self.lineargh = nn.Linear(embed_dim, embed_dim)
        self.lineargdh = nn.Linear(embed_dim, embed_dim*2)
        # 3. Refine Encoders
        self.landencoder1 = nn.ModuleList([
            Block(dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.landencoder2 = nn.ModuleList([
            Block(dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        # 4. Stage 2 Output Head
        self.outLandmarks = oLandmarkPredictor(input_channels=embed_dim*2, key_nums=self.fm_tnums)

        # Loss
        self.match_loss = matchLandmarkLoss()
        self.fixed_loss = fixedLandmarkLoss()

        self._init_weights(self.predlandmarks)
        self._init_weights(self.outLandmarks)
        self._init_weights(self.rel_pos_encoder)  # Init new module

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        pos_embed1 = PositionalEncoding(self.encoder_embed.shape[1], self.encoder_embed.shape[2], self.device)
        self.encoder_embed.data.copy_(pos_embed1.float().unsqueeze(0))

    def forward(self, p_points, trans_mask):
      

        # 返回: (S1_Fixed, S1_Match, Heatmap, S2_Fixed, S2_Match)
        return fixedlandmarks, matchLandmarks, heat_out, flandmarks_refine, mLandmarks_refine
