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
# 1. 相对位置编码器 (Relative Pos Encoder)
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

class RotaryRelativePosEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, delta_xyz):
        # 实现旋转等变编码（需结合频率编码）
        freqs = torch.pow(1000, -2 * (torch.arange(self.dim//6, device=delta_xyz.device).float() / (self.dim//6)))
        emb = delta_xyz[..., None] * freqs  # (..., 3, D//6)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (..., D)
# ==========================================
# 2. KNN 搜索
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

    # 取最近的 k 个，largest=False 表示取最小距离
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (B, M, k)
    return idx


# ==========================================
# 3. 几何感知特征提取
# ==========================================
def get_local_feature_with_geometry(init_coords, input_pos, input_feat, idx, pos_encoder):
    """
    提取局部特征，并注入相对位置信息
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

    # 4. 计算相对坐标
    delta_xyz = neighbor_xyz - init_coords.unsqueeze(2)

    # 5. 编码相对位置
    delta_feat = pos_encoder(delta_xyz)

    # 6. 特征融合 (语义 + 几何)
    combined_feat = torch.cat([neighbor_feat, delta_feat], dim=-1)

    return combined_feat

class Block3DAttn(nn.Module):
    """在原有Block基础上新增3D空间注意力"""
    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=qkv_bias)
        # 新增3D空间注意力：基于点云坐标的空间相似度
        self.d_attn = nn.Sequential(
            nn.Linear(3, dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(dim//2, 1)
        )
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(dim*mlp_ratio), dim)
        )

    def forward(self, x, pos):
        """
        x: (B, N, dim) - 特征
        pos: (B, N, 3) - 3D点云坐标（用于计算空间注意力）
        """
        # 原有自注意力
        attn_x = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + attn_x
        # 新增3D空间注意力
        spatial_w = self.d_attn(pos)  # (B, N, 1)
        x = x * spatial_w
        # 原有MLP
        x = x + self.mlp(self.norm2(x))
        return x
# ==========================================
# 4. 预测头
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


class SegRefinement(nn.Module):
    """
    专门用于分割任务的特征提取模块
    使用 Residual MLP 结构，增加深度但不改变维度
    """

    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim

        # 第一个残差块：提取形状上下文
        self.res1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 使用 LayerNorm 稳定梯度
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim)  # 再次 Norm
        )

        # 第二个残差块：细化边界
        self.res2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, in_dim)
        )

        # 最后的分类层
        self.cls_head = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: (B, N, C)

        # Residual Block 1
        res = self.res1(x)
        x = x + res  # 残差连接：保留原始语义信息，叠加形状信息

        # Residual Block 2
        res = self.res2(x)
        x_seg_feat = x + res  # (B, N, C) -> 这是专门属于分割的高级特征

        # Logits
        logits = self.cls_head(x_seg_feat)

        return logits#, x_seg_feat
# ==========================================
# 5. 主模型 ToothLandmark (完整版)
# ==========================================
class ToothLandmark(nn.Module):
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=2):
        super(ToothLandmark, self).__init__()

        self.fxid_tnums = config.fxid_tnums
        self.match_tnums = config.match_tnums
        self.fm_tnums = config.fxid_tnums + config.match_tnums
        self.knn_nums = config.knn_nums

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Point Cloud Encoder ---
        self.linear1 = nn.Linear(3, embed_dim)
        self.linear2 = nn.Linear(3, embed_dim)

        norm_layer = partial(RMSNorm, eps=1e-6)

        # Backbone Blocks
        self.Tsencoders = nn.ModuleList([
            Block(dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        # self.Tsencoders = nn.ModuleList([
        #     Block3DAttn(dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, qkv_bias=True,
        #                 norm_layer=norm_layer)
        #     for i in range(depth)])

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
        self.s1_refinement = nn.Linear(embed_dim * 2, embed_dim)
        self.s2_refinement = nn.Linear(embed_dim * 2, embed_dim)  # 可选，当前未使用

        # [NEW] 分割头 (Segmentation Head)
        # 输入维度: embed_dim (来自 s1_refinement)
        # 输出维度: 1 (Logits, 表示属于牙齿的概率)
        self.seg_head = SegRefinement(in_dim=embed_dim)

        self.linearMap1 = nn.Linear(embed_dim, embed_dim * 2)

        self.linearh = nn.Linear(1, embed_dim)
        self.heat_out = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(embed_dim, 1))

        self.predlandmarks = LandmarkPredictor(input_channels=embed_dim * 2,
                                               key_nums=self.fm_tnums)

        # --- Stage 2 Modules ---
        # 1. 相对位置编码器
        self.rel_pos_encoder = RelativePosEncoder(in_dim=3, out_dim=embed_dim)

        # 2. 降维映射
        self.lineargh = nn.Linear(embed_dim*2, embed_dim)
        self.lineargdh = nn.Linear(embed_dim, embed_dim * 2)

        # 3. Refine Encoders
        self.landencoder1 = nn.ModuleList([
            Block(dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        # 4. Stage 2 Output Head
        self.outLandmarks = oLandmarkPredictor(input_channels=embed_dim * 2, key_nums=self.fm_tnums)

        # Loss (Keeping references)
        self.match_loss = matchLandmarkLoss()
        self.fixed_loss = fixedLandmarkLoss()

        # Init Weights
        self._init_weights(self.predlandmarks)
        self._init_weights(self.outLandmarks)
        self._init_weights(self.rel_pos_encoder)
        self._init_weights(self.seg_head)  # Init new head

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
        TB, TN, N, C = p_points.shape
        NM = N // config.Tpoints

        # 数据准备
        input_pos = p_points[..., :3].clone()
        trans_mask = trans_mask > 0

        # 中心化处理
        cp = torch.mean(p_points[..., :3], dim=-2)
        if self.training:
            noise = (torch.randn(TB * 1, TN, 3).cuda().float() * 1.0)
            cp = cp + noise

        p_points_centered = p_points[..., :3] - cp.unsqueeze(dim=-2)

        # Encoder Forward
        points = self.linear1(p_points_centered)
        cpoints = self.linear2(cp)

        points = points.reshape(TB * TN, N, -1)  # Flatten
        points = points.reshape(TB * TN *NM, N//NM, -1)  # Flatten

        teeths = []
        for blk in self.Tsencoders:
            points = blk(points)
            teeths.append(points)


        for blk in self.Tcencoders:
            cpoints = blk(cpoints, trans_mask)

        x = teeths[-1]  # (TB*TN, N, embed_dim)
        x = self.linearB(x)

        condtion = cpoints.reshape(TB * TN, -1).unsqueeze(dim=1).repeat(NM, 1, 1)

        xxin = []
        for blk in self.trans_:
            x = blk(x, condtion)
            xxin.append(x)

        # Heatmap Branch
        heatMap = self.heat_out(xxin[-1].reshape(TB * TN, N, -1))
        heat_out = heatMap.reshape(TB, TN, config.sam_points, 1).sigmoid()

        # Feature Construction for Stage 1
        linearh = self.linearh(heatMap)
        x_combined = torch.cat([linearh, torch.cat(xxin[-1:], dim=-1).reshape(TB * TN, N, -1)], dim=-1)

        # Point-wise features (TB*TN, N, embed_dim)
        s1_point_features = self.s1_refinement(x_combined)

        # =======================================================
        # [NEW] Segmentation & Feature Gating
        # =======================================================
        # 1. 预测分割 logits (TB*TN, N, 1)
        seg_logits = self.seg_head(s1_point_features)

        # 2. 计算概率 (用于 Gating)
        seg_score = torch.sigmoid(seg_logits)


        s1_point_features_gated = s1_point_features

        # =======================================================

        x_global = torch.max(s1_point_features_gated, dim=1)[0]
        x_global = self.linearMap1(x_global)

        # Coarse Prediction
        predlandmarks = self.predlandmarks(x_global).reshape(TB, TN, self.fm_tnums, -1)

        # 恢复绝对坐标
        predlandmarks_abs = predlandmarks.clone()
        predlandmarks_abs[..., :3] = predlandmarks[..., :3] + cp.unsqueeze(dim=-2)

        fixedlandmarks = predlandmarks_abs[..., :config.fxid_tnums, :].reshape(TB, TN, -1)
        matchLandmarks = predlandmarks_abs[..., config.fxid_tnums:, :].reshape(TB, TN, -1)

        # =========================================================
        # Stage 2: Local Refinement (Improved with Gated Features)
        # =========================================================

        # 1. 准备 Query 坐标 (Detach)
        init_coords = predlandmarks_abs[..., :3].reshape(TB * TN, self.fm_tnums, 3).detach()
        input_pos_flat = input_pos.reshape(TB * TN, config.sam_points, 3)

        # 2. KNN 搜索
        idx = knn_cross(init_coords, input_pos_flat, k=self.knn_nums)

        # 3. 几何特征聚合
        # 注意：这里传入的是 s1_point_features_gated
        # 这样 Stage 2 在聚合局部特征时，会“看到”被分割掩码强化过的牙齿特征
        land_fea = get_local_feature_with_geometry(
            init_coords,
            input_pos_flat,
            s1_point_features_gated,
            idx,
            self.rel_pos_encoder
        )

        # 4. 降维适应 Transformer
        land_fea = self.lineargh(land_fea)  # (TB*TN, M, embed_dim)

        # 6. Refine Transformers
        land_fea = land_fea.reshape(TB * TN * self.fm_tnums, self.knn_nums, -1)
        for blk in self.landencoder1:
            land_fea = blk(land_fea)

        land_fea = land_fea.reshape(TB * TN, self.fm_tnums, self.knn_nums, -1).max(dim=-2)[0]
        land_fea = self.lineargdh(land_fea)


        fland_fea = land_fea[:, :config.fxid_tnums, :]
        mland_fea = land_fea[:, config.fxid_tnums:, :]

        # 7. Predict Offset
        land_offsets = self.outLandmarks(fland_fea, mland_fea)

        # 8. Final Coordinates
        land_offsets[..., :3] = land_offsets[..., :3] + init_coords.reshape(TB * TN, self.fm_tnums, 3)

        final_coords = land_offsets.reshape(TB, TN, self.fm_tnums, -1)

        flandmarks_refine = final_coords[..., :config.fxid_tnums, :].reshape(TB, TN, -1)
        mLandmarks_refine = final_coords[..., config.fxid_tnums:, :].reshape(TB, TN, -1)


        # 恢复 Seg Logits 形状以便 Loss 计算 (TB, TN, N, 1)
        seg_score = seg_score.reshape(TB, TN, N, 1)

        # 返回: (S1_Fixed, S1_Match, Heatmap, S2_Fixed, S2_Match, Center, S1_Coords, [NEW]Seg_Logits)
        return fixedlandmarks, matchLandmarks, heat_out, flandmarks_refine, mLandmarks_refine, seg_score