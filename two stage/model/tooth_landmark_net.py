
from functools import partial
import torch

import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
from model.pos_embed import PositionalEncoding
from model.jitblock import JiTBlock, RMSNorm
from model.transformer import Transformer
from model.loss import matchLandmarkLoss, fixedLandmarkLoss
##############################################################
from config import config

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
        idx = knn(x[:, 0:3], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idxx = idx + idx_base

    idxx = idxx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idxx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()



    return feature, idx   # (batch_size, 2*num_dims, num_points, k)

class fixedLandmarkPredictor(nn.Module):
    def __init__(self, input_channels, share_channels, num_teeth=16, key_nums=5):
        """
        Args:
            input_channels (int): 输入特征的通道数 C
            num_teeth (int): 牙齿数量，默认为 16
            out_dims_list (list): 长度为 16 的列表，包含每颗牙齿对应的关键点坐标输出维度 (如 x,y 坐标总数)
        """
        super(fixedLandmarkPredictor, self).__init__()
        self.num_heads = key_nums
        self.num_teeth = num_teeth
        # 确保通道数能被 17 整除，或者根据你的需求手动指定分块大小
        self.chunk_size = (input_channels -share_channels) // num_teeth

        # 定义 16 个独立的回归头
        # 每个回归头包含两个全连接层

        # 每个分支的输入 = 专属特征(chunk_size) + 共享特征(chunk_size)
        in_features = input_channels#self.chunk_size + share_channels
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.LeakyReLU(),
                nn.Linear(in_features // 2, 1 * (3 + 3 + 1))
            ) for _ in range(key_nums)
        ])

    def forward(self, x):
        """
        假设 x 的维度是 [Batch, num_heads, in_features]
        或者针对不同的场景进行遍历
        """
        results = []
        for i in range(self.num_heads):
            # 每个 head 处理对应的一份特征
            results.append(self.heads[i](x))

        # 将结果拼接回去 [Batch, num_heads, out_features * 7]
        return torch.cat(results, dim=-1)

class matchLandmarkPredictor(nn.Module):
    def __init__(self, input_channels, share_channels, num_teeth=16, key_nums=5):
        """
        Args:
            input_channels (int): 输入特征的通道数 C
            num_teeth (int): 牙齿数量，默认为 16
            out_dims_list (list): 长度为 16 的列表，包含每颗牙齿对应的关键点坐标输出维度 (如 x,y 坐标总数)
        """
        super(matchLandmarkPredictor, self).__init__()
        self.num_heads = key_nums
        self.num_teeth = num_teeth
        # 确保通道数能被 17 整除，或者根据你的需求手动指定分块大小
        self.chunk_size = (input_channels -share_channels) // num_teeth

        # 每个分支的输入 = 专属特征(chunk_size) + 共享特征(chunk_size)
        in_features = input_channels#self.chunk_size + share_channels
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.LeakyReLU(),
                nn.Linear(in_features // 2, 1 * (3 + 3 + 1))
            ) for _ in range(key_nums)
        ])

    def forward(self, x):
        """
        假设 x 的维度是 [Batch, num_heads, in_features]
        或者针对不同的场景进行遍历
        """
        results = []
        for i in range(self.num_heads):
            # 每个 head 处理对应的一份特征
            results.append(self.heads[i](x))

        # 将结果拼接回去 [Batch, num_heads, out_features * 7]
        return torch.cat(results, dim=-1)
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

class ToothLandmark(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, embed_dim=256, depth=4, num_heads=4,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(ToothLandmark, self).__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.linear1 = nn.Linear(3, embed_dim)
        self.geo_res = GeometricResidual(embed_dim)
        self.linear2 = nn.Linear(3, embed_dim)

        self.locaConv = nn.Sequential(nn.Conv2d(6*2, embed_dim, kernel_size=1, bias=False),
                                   nn.InstanceNorm2d(embed_dim),
                                   nn.LeakyReLU(negative_slope=0.2))


        norm_layer= partial(RMSNorm, eps=1e-6)#partial(nn.LayerNorm, eps=1e-6)
        self.Tsencoders = nn.ModuleList([
            Block(dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.Tcurencoders = nn.ModuleList([
            Block(dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.linearB = nn.Linear(embed_dim, embed_dim)

        self.Tcencoders = nn.ModuleList([
            Transformer(dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.datap = nn.Parameter(torch.zeros(1, config.tooth_nums, 1, embed_dim)).cuda().float()
        self.w1 = nn.Parameter(torch.zeros(1, config.sam_points, 1)).cuda().float()

        self.encoder_embed = nn.Parameter(torch.zeros(1, config.tooth_nums, embed_dim), requires_grad=False)

        self.pos_embed = RelativePosEncoder(in_dim=3, out_dim=embed_dim)

        self.trans_ = nn.ModuleList([
            JiTBlock(hidden_size=embed_dim, num_heads=embed_dim // 64, mlp_ratio=mlp_ratio, attn_drop=0.0, proj_drop=0.0)
            for i in range(depth)])
        input_channels = embed_dim-64 + 64*config.tooth_nums
        self.linearMap = nn.Linear(embed_dim*2, embed_dim*4)
        self.linearMap1 = nn.Linear(embed_dim*4, embed_dim*4)
        self.norm = nn.LayerNorm(embed_dim*4)

        self.linearh= nn.Linear(1, embed_dim)
        self.heat_out = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(embed_dim, 1))

        self.fixedlandmarks1 = fixedLandmarkPredictor(input_channels=embed_dim*4, share_channels=embed_dim - 64,
                                                        num_teeth=16, key_nums=config.fxid_tnums)

        self.matchLandmarks = matchLandmarkPredictor(input_channels=embed_dim*4, share_channels=embed_dim-64,
                                                        num_teeth=16, key_nums=config.match_tnums)

        self. match_loss = matchLandmarkLoss()
        self.fixed_loss = fixedLandmarkLoss()
        # self.initialize_weights()

        self._init_weights(self.matchLandmarks)
        self._init_weights(self.fixedlandmarks1)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        # initialization

        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed1 = PositionalEncoding(self.encoder_embed.shape[1], self.encoder_embed.shape[2], self.device)
        self.encoder_embed.data.copy_(pos_embed1.float().unsqueeze(0))

    def forward(self, p_points,  trans_mask):

        trans_mask = trans_mask > 0
        TB, TN, N, C = p_points.shape

        cp = torch.mean(p_points[..., :3], dim=-2)
        if self.training:
            noise = (torch.randn(TB*1, TN, 3).cuda().float()*1.0)#.clip(-3, 3)
            cp = cp + noise#torch.cat([cp, cp + noise], dim=0)
            # p_points = p_points.expand(1+TB, -1, -1, -1)

        TB, TN, N, C = p_points.shape

        p_points = p_points[..., :3] - cp.unsqueeze(dim=-2)
        points_xyz = p_points.reshape(TB * TN, N, -1)

        # 3. 线性映射 + 几何残差注入
        # x_base 是原始的语义 Embedding
        points = self.linear1(points_xyz)
        # points = x_base + self.geo_res(points_xyz)

        cpoints = self.linear2(cp)


        _, N, C = points.shape
        points = points.reshape(TB* TN, N, C)
        #curev = curev.reshape(TB * TN, N, C)

        teeths = []
        for blk in self.Tsencoders:
            points = blk(points)
            teeths.append(points)

        # curevs = []
        # for blk in self.Tcurencoders:
        #     curev = blk(curev)
        #     curevs.append(curev)


        for blk in self.Tcencoders:
            cpoints = blk(cpoints, trans_mask)

        x = teeths[-1]#torch.stack(teeths).sum(dim=0)#
        #x = torch.cat([teeths[-1], curevs[-1]], dim=-1)
        x = self.linearB(x)
        condtion = cpoints.reshape(TB* TN, C).unsqueeze(dim=1)
        #condtion = (cpoints + self.encoder_embed.expand(TB, -1, -1)).reshape(TB* TN, C).unsqueeze(dim=1)

        pos_embed = self.pos_embed(p_points.reshape(TB* TN, N, -1))

        x = x + pos_embed
        xxin =[]
        for blk in self.trans_:
            x = blk(x, condtion)
            xxin.append(x)
        #xxin = torch.max(torch.cat(xxin, dim=-1), dim=1, keepdim=True)[0]

        heat_out = self.heat_out(xxin[-1])

        linearh = self.linearh(heat_out)
        xxin = torch.cat([linearh, torch.cat(xxin[-1:], dim=-1)], dim=-1)
        x = self.linearMap(xxin)

        x = torch.max(x, dim=1, keepdim=True)[0]


        x = self.linearMap1(x)


        # restored_x = torch.zeros((TN, x.size(1), x.size(2)), device = x.device, dtype = x.dtype)
        # indices = torch.nonzero(trans_mask.squeeze() > 0).squeeze()
        # # 3. 将计算好的特征 x 填回到对应位置
        # restored_x[indices] = x

        restored_x = x

        fixedlandmarks = self.fixedlandmarks1(restored_x)
        matchLandmarks = self.matchLandmarks(restored_x)

        fixedlandmarks = fixedlandmarks.reshape(TB, TN, config.fxid_tnums, 7)
        fixedlandmarks[..., :3] = fixedlandmarks[..., :3] +cp.unsqueeze(dim=-2)
        fixedlandmarks = fixedlandmarks.reshape(TB, TN, -1)

        matchLandmarks = matchLandmarks.reshape(TB, TN, config.match_tnums, 7)
        matchLandmarks[..., :3] = matchLandmarks[..., :3] +cp.unsqueeze(dim=-2)
        matchLandmarks = matchLandmarks.reshape(TB, TN, -1)

        return fixedlandmarks, matchLandmarks, heat_out.reshape(TB, TN, config.sam_points, 1).sigmoid()











