import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block

# 保持原始导入结构不变
from model.pos_embed import PositionalEncoding
from model.jitblock import JiTBlock, RMSNorm
from model.transformer import Transformer
from model.loss import matchLandmarkLoss, fixedLandmarkLoss
from config import config

# 额外导入FAISS（KNN加速，按需使用）
try:
    import faiss
    import numpy as np

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ======================== 核心模块 ========================
class RotaryRelativePosEncoder(nn.Module):
    """旋转等变相对位置编码器，保留3D空间旋转不变性"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.freqs = nn.Parameter(
            torch.pow(10000, -2 * (torch.arange(self.dim // 6).float() / (self.dim // 6))),
            requires_grad=False
        )

    def forward(self, delta_xyz):
        device = delta_xyz.device
        freqs = self.freqs.to(device)

        # 频率编码
        emb = delta_xyz[..., None] * freqs  # (..., 3, D//6)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (..., 3, D//3)
        emb = emb.flatten(-2)  # (..., D)

        # 确保维度匹配
        if emb.shape[-1] < self.dim:
            pad = torch.zeros(*emb.shape[:-1], self.dim - emb.shape[-1], device=device)
            emb = torch.cat([emb, pad], dim=-1)

        return emb


def knn_cross(init_coords, input_pos, k):
    """
    优化版KNN搜索（兼容FAISS加速）
    init_coords: (B, M, 3) - 中心点
    input_pos:   (B, N, 3) - 源点云
    """
    # 使用FAISS加速（如果可用）
    if FAISS_AVAILABLE:
        B, M, _ = init_coords.shape
        _, N, _ = input_pos.shape
        k = min(k, N)  # 安全检查：防止k大于点云数量

        # 转换为FAISS支持的格式
        init_np = init_coords.detach().cpu().numpy().astype(np.float32)
        input_np = input_pos.detach().cpu().numpy().astype(np.float32)

        idx_batch = []
        for b in range(B):
            index = faiss.IndexFlatL2(3)
            index.add(input_np[b])
            _, idx = index.search(init_np[b], k)
            idx_batch.append(torch.from_numpy(idx).to(init_coords.device))

        return torch.stack(idx_batch, dim=0)
    else:
        # 原始PyTorch实现（添加安全检查）
        dist_init = torch.sum(init_coords ** 2, dim=-1, keepdim=True)  # (B, M, 1)
        dist_input = torch.sum(input_pos ** 2, dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, N)
        inner = torch.matmul(init_coords, input_pos.transpose(1, 2))  # (B, M, N)

        pairwise_distance = dist_init + dist_input - 2 * inner
        # 处理无效距离（如NaN）
        pairwise_distance = torch.nan_to_num(pairwise_distance, nan=1e9, posinf=1e9)
        # 安全检查k值
        k = min(k, input_pos.shape[1])
        # 取最近的 k 个
        idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (B, M, k)
        return idx


def get_local_feature_with_geometry(init_coords, input_pos, input_feat, idx, pos_encoder):
    """优化内存占用的几何特征聚合"""
    TN, M, K = idx.shape
    C = input_feat.shape[-1]
    device = input_feat.device

    # 展平索引减少内存拷贝（优化核心）
    idx_flat = idx.reshape(TN, -1)  # (TN, M*K)
    batch_indices = torch.arange(TN, device=device).view(TN, 1).expand(TN, M * K)

    # Gather邻域特征和坐标
    neighbor_feat = input_feat[batch_indices, idx_flat, :].reshape(TN, M, K, C)
    neighbor_xyz = input_pos[batch_indices, idx_flat, :].reshape(TN, M, K, 3)

    # 计算相对坐标并编码
    delta_xyz = neighbor_xyz - init_coords.unsqueeze(2)
    delta_feat = pos_encoder(delta_xyz)

    # 特征融合
    combined_feat = torch.cat([neighbor_feat, delta_feat], dim=-1)

    return combined_feat


def multi_scale_feature_fusion(features: list[torch.Tensor]) -> torch.Tensor:
    """多尺度特征融合，利用不同层级的特征信息"""
    weights = nn.Parameter(torch.ones(len(features))).to(features[0].device)
    weights = F.softmax(weights, dim=0)

    fused_feat = 0
    for i, feat in enumerate(features):
        fused_feat += weights[i] * feat

    return fused_feat


# ======================== 预测头 ========================
class LandmarkPredictor(nn.Module):
    """带Dropout的关键点预测头，防止过拟合"""

    def __init__(self, input_channels, key_nums=5):
        super().__init__()
        self.num_heads = key_nums
        in_features = input_channels
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.LeakyReLU(config.leaky_relu_slope if hasattr(config, 'leaky_relu_slope') else 0.2),
                nn.Dropout(config.dropout_rate if hasattr(config, 'dropout_rate') else 0.1),
                nn.Linear(in_features // 2, config.prediction_dim if hasattr(config, 'prediction_dim') else 7)
            ) for _ in range(key_nums)
        ])

    def forward(self, x):
        results = [self.heads[i](x) for i in range(self.num_heads)]
        return torch.stack(results, dim=-2)


class oLandmarkPredictor(nn.Module):
    """Stage2 Offset预测头（带Dropout）"""

    def __init__(self, input_channels, key_nums=5):
        super().__init__()
        in_features = input_channels
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.LeakyReLU(config.leaky_relu_slope if hasattr(config, 'leaky_relu_slope') else 0.2),
                nn.Dropout(config.dropout_rate if hasattr(config, 'dropout_rate') else 0.1),
                nn.Linear(in_features // 2, config.prediction_dim if hasattr(config, 'prediction_dim') else 7)
            ) for _ in range(2)  # 0: fixed, 1: match
        ])

    def forward(self, x, x1):
        return torch.cat([self.heads[0](x), self.heads[1](x1)], dim=-2)


class SegRefinement(nn.Module):
    """增强版分割分支（添加注意力机制）"""

    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        dropout_rate = config.dropout_rate if hasattr(config, 'dropout_rate') else 0.1
        leaky_slope = config.leaky_relu_slope if hasattr(config, 'leaky_relu_slope') else 0.2

        # 局部注意力模块（增强核心）
        self.attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )

        # 残差块
        self.res1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim)
        )

        self.res2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(leaky_slope),
            nn.Linear(hidden_dim, in_dim)
        )

        self.cls_head = nn.Linear(in_dim, 1)

    def forward(self, x):
        # 注意力增强特征
        x_attn, _ = self.attn(x, x, x)
        x = x + 0.1 * x_attn  # 注意力残差

        # 残差连接
        res = self.res1(x)
        x = x + res

        res = self.res2(x)
        x_seg_feat = x + res

        return self.cls_head(x_seg_feat)


# ======================== 主模型 ========================
class ToothLandmark(nn.Module):
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=2):
        super(ToothLandmark, self).__init__()

        # 保留原始config引用
        self.fxid_tnums = config.fxid_tnums
        self.match_tnums = config.match_tnums
        self.fm_tnums = config.fxid_tnums + config.match_tnums
        self.knn_nums = config.knn_nums if hasattr(config, 'knn_nums') else 32

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Point Cloud Encoder ---
        self.linear1 = nn.Linear(3, embed_dim)
        self.linear2 = nn.Linear(3, embed_dim)

        norm_layer = partial(RMSNorm, eps=1e-6)

        # 复用Block创建逻辑（减少冗余）
        def create_block():
            return Block(
                dim=embed_dim,
                num_heads=embed_dim // 64,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )

        # Backbone Blocks
        self.Tsencoders = nn.ModuleList([create_block() for _ in range(depth)])
        self.Tcurencoders = nn.ModuleList([create_block() for _ in range(depth)])

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
        # 移除未使用的s2_refinement
        # self.s2_refinement = nn.Linear(embed_dim * 2, embed_dim)

        # 分割头（增强版）
        self.seg_head = SegRefinement(in_dim=embed_dim)

        self.linearMap1 = nn.Linear(embed_dim, embed_dim * 2)

        self.linearh = nn.Linear(1, embed_dim)
        leaky_slope = config.leaky_relu_slope if hasattr(config, 'leaky_relu_slope') else 0.2
        self.heat_out = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                      nn.LeakyReLU(leaky_slope),
                                      nn.Linear(embed_dim, 1))

        self.predlandmarks = LandmarkPredictor(input_channels=embed_dim * 2,
                                               key_nums=self.fm_tnums)

        # --- Stage 2 Modules ---
        # 1. 旋转等变相对位置编码器
        self.rel_pos_encoder = RotaryRelativePosEncoder(dim=embed_dim)

        # 2. 降维映射
        self.lineargh = nn.Linear(embed_dim * 2, embed_dim)
        self.lineargdh = nn.Linear(embed_dim, embed_dim * 2)

        # 3. Refine Encoders
        self.landencoder1 = nn.ModuleList([create_block() for _ in range(depth)])

        # 4. Stage 2 Output Head
        self.outLandmarks = oLandmarkPredictor(input_channels=embed_dim * 2, key_nums=self.fm_tnums)

        # Loss (保持原始引用)
        self.match_loss = matchLandmarkLoss()
        self.fixed_loss = fixedLandmarkLoss()

        # Init Weights
        self._init_weights(self.predlandmarks)
        self._init_weights(self.outLandmarks)
        self._init_weights(self.rel_pos_encoder)
        self._init_weights(self.seg_head)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, RMSNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        pos_embed1 = PositionalEncoding(self.encoder_embed.shape[1], self.encoder_embed.shape[2], self.device)
        self.encoder_embed.data.copy_(pos_embed1.float().unsqueeze(0))

    def center_and_normalize(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """鲁棒的点云中心化+归一化（增强核心）"""
        center = torch.mean(points, dim=-2, keepdim=True)  # (B, 1, 3)
        centered = points - center

        # 归一化到单位球（提升鲁棒性）
        max_dist = torch.max(torch.norm(centered, dim=-1, keepdim=True), dim=-2, keepdim=True)[0]
        scale = torch.clamp(max_dist, min=1e-6)  # 防止除零
        normalized = centered / scale

        return normalized, center.squeeze(-2), scale.squeeze(-2)

    def forward(self, p_points, trans_mask):
       

        # 返回原始格式
        return fixedlandmarks, matchLandmarks, heat_out, flandmarks_refine, mLandmarks_refine, seg_score

    # ======================== 新增：训练/验证工具函数 ========================
    def train_step(self, optimizer, p_points, trans_mask, targets):
        """单步训练（带梯度裁剪）"""
        self.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = self(p_points, trans_mask)
        fixed_refine, match_refine, heat_out, seg_score = outputs[3], outputs[4], outputs[2], outputs[5]

        # 计算损失
        loss_fixed = self.fixed_loss(fixed_refine, targets['fixed'])
        loss_match = self.match_loss(match_refine, targets['match'])
        loss_seg = F.binary_cross_entropy(seg_score, targets['seg_mask'])
        loss_heatmap = F.binary_cross_entropy(heat_out, targets['heatmap'])

        total_loss = loss_fixed + loss_match + 0.1 * loss_seg + 0.01 * loss_heatmap

        # 反向传播 + 梯度裁剪（防止梯度爆炸）
        total_loss.backward()
        grad_clip_norm = config.grad_clip_norm if hasattr(config, 'grad_clip_norm') else 1.0
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'loss_fixed': loss_fixed.item(),
            'loss_match': loss_match.item(),
            'loss_seg': loss_seg.item(),
            'loss_heatmap': loss_heatmap.item()
        }

    @torch.no_grad()
    def validate(self, p_points, trans_mask, targets):
        """验证步骤（无梯度）"""
        self.eval()
        outputs = self(p_points, trans_mask)
        fixed_refine, match_refine = outputs[3], outputs[4]

        # 计算损失和指标
        loss_fixed = self.fixed_loss(fixed_refine, targets['fixed'])
        loss_match = self.match_loss(match_refine, targets['match'])
        fixed_error = torch.mean(torch.norm(fixed_refine - targets['fixed'], dim=-1))
        match_error = torch.mean(torch.norm(match_refine - targets['match'], dim=-1))

        return {
            'loss_fixed': loss_fixed.item(),
            'loss_match': loss_match.item(),
            'fixed_error': fixed_error.item(),
            'match_error': match_error.item()
        }

    def save_checkpoint(self, path: str, epoch: int, optimizer=None):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, optimizer=None):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']


# ======================== 测试代码（可选） ========================
if __name__ == "__main__":
    # 初始化模型
    model = ToothLandmark()
    model.to(model.device)
    model.initialize_weights()

    # 构造测试数据
    TB, TN, N, C = 2, config.tooth_nums, config.sam_points, 3
    p_points = torch.randn(TB, TN, N, C).to(model.device)
    trans_mask = torch.randn(TB, TN, TN).to(model.device)

    # 测试前向传播
    with torch.no_grad():
        outputs = model(p_points, trans_mask)
        print("模型输出维度：")
        print(f"fixedlandmarks: {outputs[0].shape}")
        print(f"matchLandmarks: {outputs[1].shape}")
        print(f"heat_out: {outputs[2].shape}")
        print(f"flandmarks_refine: {outputs[3].shape}")
        print(f"mLandmarks_refine: {outputs[4].shape}")
        print(f"seg_score: {outputs[5].shape}")
