import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# ==========================================
# 0. 模拟配置文件 (请替换为你自己的 config)
# ==========================================
class DummyConfig:
    tnums = 16  # 类别数 (例如上颌/下颌16颗牙)
    kpnums = 7  # 每颗牙齿的关键点数量
    landPorder = ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'Cusp', 'Cenp']  # 示例关键点名称


cfg = DummyConfig()


# ==========================================
# 1. 基础工具函数与基础网络模块
# ==========================================
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


class LocalGeometricRefiner(nn.Module):
    def __init__(self, in_dim, out_dim, groups=8):
        super().__init__()
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
        feat_max = feat.max(dim=-1)[0]
        feat_avg = feat.mean(dim=-1)
        feat_pooled = torch.cat([feat_max, feat_avg], dim=1)
        feat_pooled = feat_pooled.permute(0, 2, 1).contiguous()
        out = self.mlp2(feat_pooled)
        return out


# ==========================================
# 2. 第一阶段网络 (ToothLandmark)
# ==========================================
class ToothLandmark(nn.Module):
    def __init__(self, classnums=16, k=20, topk=20):
        super(ToothLandmark, self).__init__()
        self.k = k
        self.maskk = 40
        self.topk = topk
        self.dim = 64
        self.classnums = classnums
        self.kpnums = cfg.kpnums
        groups = 8

        self.edge_conv1 = EdgeConvBlock(3, self.dim, k=k, groups=groups)
        self.edge_conv2 = EdgeConvBlock(self.dim, self.dim * 2, k=k, groups=groups)
        self.edge_conv3 = EdgeConvBlock(self.dim * 2, self.dim * 4, k=k, groups=groups)
        self.edge_conv4 = EdgeConvBlock(self.dim * 4, self.dim * 4, k=k, groups=groups)

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

        self.heatmap_head = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, 1, kernel_size=1))

        self.offset_head = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, 3, kernel_size=1))

        self.class_t = nn.Sequential(
            nn.Conv1d(self.dim * 16, self.dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, self.classnums, kernel_size=1))

        self.k_reg = 128
        self.pos_embed = nn.Sequential(
            nn.Conv2d(3, self.dim * 2, kernel_size=1),
            nn.GroupNorm(self.dim * 2 // 32, self.dim * 2),
            nn.LeakyReLU(0.2)
        )

        self.mapliner = nn.Sequential(
            nn.Conv2d(self.dim * 16, self.dim * 8, kernel_size=1),
            nn.GroupNorm(8, self.dim * 8),
            nn.LeakyReLU(0.2)
        )

        refiner_in_dim = self.dim * 8 + self.dim * 2
        self.local_refiner = LocalGeometricRefiner(in_dim=refiner_in_dim, out_dim=self.dim * 8)

        self.predconf = nn.Sequential(
            nn.Linear(self.dim * 8, self.dim * 4),
            nn.LayerNorm(self.dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim * 4, 1))

        self.predland1 = nn.Sequential(
            nn.Linear(self.dim * 8, self.dim * 4),
            nn.LayerNorm(self.dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim * 4, 3))

    def forward(self, x):
        

        return heatmap, offset_raw, cls_scores, init_conf, init_land, shifted_pos_all, final_features


# ==========================================
# 3. 辅助截取函数 (NMS & 局部 KNN)
# ==========================================
def nms_3d(proposals, scores, radius=3.0, max_keep=16, score_thresh=0.1):

    B, N, _ = proposals.shape
    # 提前准备好掩码索引，避免后面 gather 报错

    for b in range(B):
        b_proposals = proposals[b]
        b_scores = scores[b]

        # 1. 先解决你说的阈值问题
        valid_mask = b_scores > score_thresh
        if not valid_mask.any():
            # 如果这一组全军覆没，直接跳过（默认填0或-1）
            continue

        # 2. 仅对达标的点进行排序
        valid_inds = torch.where(valid_mask)[0]
        sub_scores = b_scores[valid_inds]
        order = valid_inds[torch.argsort(sub_scores, descending=True)]

        b_keep = []
        while order.numel() > 0 and len(b_keep) < max_keep:
            i = order[0].item()
            b_keep.append(i)

            if order.numel() == 1:
                break

            # 计算距离并过滤
            center = b_proposals[i].unsqueeze(0)
            other_points = b_proposals[order[1:]]
            dists = torch.norm(other_points - center, dim=-1)

            # 只有距离大于 radius 的点才进入下一轮候选
            inds = torch.where(dists > radius)[0]
            order = order[inds + 1]

        # 3. 填充逻辑改进：不要复读机！
        # 如果找出来的牙齿不够，剩下的位置建议填一个特殊索引（比如-1）或者直接截断
        # 如果你后续必须用 gather，先填已有的，剩下的保持默认（例如 0）
        num_found = len(b_keep)
        keep_idx = torch.zeros((B, num_found), dtype=torch.long, device=proposals.device)

        if num_found > 0:
            keep_idx[b, :num_found] = torch.tensor(b_keep, device=proposals.device)
            # 如果你一定要填满到 max_keep，且不想重复中心点，
            # 可以考虑填入一个在坐标系之外的“虚拟点”

    return keep_idx


def get_local_crops(raw_pos, final_features, centers, k=1536):
    B, num_teeth, _ = centers.shape
    _, D, N = final_features.shape

    dists = torch.cdist(centers, raw_pos)
    _, knn_idx = torch.topk(dists, k=k, dim=-1, largest=False)

    knn_idx_pos = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    local_pos = torch.gather(raw_pos.unsqueeze(1).expand(-1, num_teeth, -1, -1), 2, knn_idx_pos)

    centers_expanded = centers.unsqueeze(2)
    relative_pos = local_pos - centers_expanded

    knn_idx_feat = knn_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    final_features_t = final_features.transpose(1, 2)
    local_feat = torch.gather(final_features_t.unsqueeze(1).expand(-1, num_teeth, -1, -1), 2, knn_idx_feat)

    relative_pos = relative_pos.view(B * num_teeth, k, 3)
    local_feat = local_feat.view(B * num_teeth, k, D)

    crop_features = torch.cat([relative_pos, local_feat], dim=-1).transpose(1, 2).contiguous()
    return crop_features, relative_pos


# ==========================================
# 4. 第二阶段：局部关键点预测与分割网络 (已强化，引入 EdgeConv + Seg 分支)
# ==========================================
class LocalLandmarkPredictor(nn.Module):
    def __init__(self, in_channels=1024 + 3, num_landmarks=6, dim=64, groups=8):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.dim = 32
        dim = 32
        # 1. 通道降维与基础特征提取
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 2. 引入 EdgeConv，加强局部微观拓扑结构的特征提取 (比纯 1x1 Conv 能力强很多)
        # 考虑到局部点云 k=1536，近邻 k 设为 16 或 20 较为合适
        self.edge_conv1 = EdgeConvBlock(dim * 4, dim * 4, k=16, groups=groups)
        self.edge_conv2 = EdgeConvBlock(dim * 4, dim * 4, k=16, groups=groups)

        # 3. 特征融合模块
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(dim * 8, dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 4. 全局上下文模块 (结合 Max Pool 和 Avg Pool)
        self.global_mlp = nn.Sequential(
            nn.Conv1d(dim * 4 * 2, dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 5. 解码融合模块
        self.decoder = nn.Sequential(
            nn.Conv1d(dim * 4 + dim * 4, dim * 8, kernel_size=1, bias=False),
            nn.GroupNorm(groups, dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- 以下为并行的三个输出分支 ---

        # 分支 A: 预测每颗牙齿的 6 通道关键点热图
        self.heatmap_head = nn.Sequential(
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, self.num_landmarks, kernel_size=1))

        # 分支 B: 预测对应关键点的 3D 偏移量
        self.offset_head = nn.Sequential(
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, self.num_landmarks * 3, kernel_size=1))

        # 分支 C (新增): 预测局部 1536 个点中，哪些属于当前的中心牙齿 (二分类)
        self.seg_head = nn.Sequential(
            nn.Conv1d(self.dim * 8, self.dim * 4, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(self.dim * 4, 1, kernel_size=1))

    def forward(self, local_crops):
        # local_crops: (B * max_teeth, in_channels, K)
        x_stem = self.stem(local_crops)

        # 在特征空间动态提取局部邻域特征
        x_edge1, idx1 = self.edge_conv1(x_stem)
        x_edge2, idx2 = self.edge_conv2(x_edge1, idx=idx1)

        # 聚合 EdgeConv 提取的局部表面特征
        x_local = torch.cat([x_edge1, x_edge2], dim=1)
        x_local = self.conv_fuse(x_local)

        # 提取全局上下文并扩维
        x_max = torch.max(x_local, dim=-1, keepdim=True)[0]
        x_avg = torch.mean(x_local, dim=-1, keepdim=True)
        x_global_concat = torch.cat([x_max, x_avg], dim=1)
        x_global = self.global_mlp(x_global_concat)
        x_global_expand = x_global.repeat(1, 1, x_local.shape[-1])

        # 将局部特征与全局特征拼接，供解码使用
        x_feat = torch.cat([x_local, x_global_expand], dim=1)
        x_feat = self.decoder(x_feat)

        # 输出层预测
        # 形状: (B*max_teeth, K, num_landmarks)
        heatmap = torch.sigmoid(self.heatmap_head(x_feat)).permute(0, 2, 1)

        # 形状: (B*max_teeth, K, num_landmarks*3) -> (B*max_teeth, K, num_landmarks, 3)
        offset_raw = self.offset_head(x_feat).permute(0, 2, 1)
        K_points = offset_raw.shape[1]
        offset_raw = offset_raw.view(-1, K_points, self.num_landmarks, 3)

        # 形状: (B*max_teeth, K) 局部牙齿的二分类概率蒙版
        seg_mask = torch.sigmoid(self.seg_head(x_feat)).squeeze(1)

        return heatmap, offset_raw, seg_mask


# ==========================================
# 5. 最终串联：端到端 Two-Stage 架构
# ==========================================
class TwoStageToothPipeline(nn.Module):
    def __init__(self, classnums=cfg.tnums, max_teeth=16, num_landmarks=cfg.kpnums, topk_stage1=20):
        super().__init__()
        self.max_teeth = max_teeth
        self.stage1 = ToothLandmark(classnums=classnums, topk=topk_stage1)
        # 注意: 实际使用时解除这里的注释以加载预训练权重
        # model_path = "./outputs/seg_land_oneheat_final.pth"
        # from main_oneheat_reg import model_initial
        # model_initial(self.stage1, model_path)

        in_dim_stage2 = 3 + (64 * 16)
        self.stage2_landmark_head = LocalLandmarkPredictor(
            in_channels=in_dim_stage2,
            num_landmarks=num_landmarks
        )

    def forward(self, x):
  


        return heatmap, offset_raw, cls_scores, init_land, init_conf, shifted_pos_all,twoheatmap, twooffset_raw, twoseg_mask, final_centers, relative_pos
        # 增加返回 twoseg_mask，供计算 DiceLoss / FocalLoss 使用
        return {"onestage":[heatmap, offset_raw, cls_scores, init_land, init_conf, shifted_pos_all],
               "twostage":[twoheatmap, twooffset_raw, twoseg_mask, final_centers, relative_pos]}

if __name__ == "__main__":

    model = ToothLandmark()#

    heatt_loss = ToothHeatmapLoss()
    off_loss = torch.nn.SmoothL1Loss()
    match_loss = OneMultiInstanceMatchLoss()
    criterion = nn.CrossEntropyLoss()

    for batch_data, heat_map, offest_map, gtcls, mask, teeth_landmarks in train_loader:
        #label_landmarks = add_gaussian_noise(label_landmarks)

        nums = nums +1
        batch_data = batch_data.cuda().float()
        heat_map = heat_map.cuda().float()
        offest_map = offest_map.squeeze(dim=-2).cuda().float()
        gtcls = gtcls.cuda().float()
        mask = mask.cuda().bool()
        optimizer.zero_grad()
        with autocast():
            preheat_map, preoff_map, cls, init_conf, init_land, shifted_pos_all = model(batch_data)


        dice_loss_ = heatt_loss(preheat_map, heat_map, mask)
        cls_loss_ = heatt_loss(cls, gtcls, gtcls.cuda().bool())

        wing_Loss_ = off_loss(preoff_map.float()[mask.expand_as(offest_map)],
                              offest_map.float()[mask.expand_as(offest_map)])

        comm_loss, coord_loss_, fm_conf_loss_, no_tooth_conf_loss_, tooth_conf_loss_ = match_loss(init_conf, init_land,
                                                                                                  shifted_pos_all,
                                                                                                  teeth_landmarks)

        loss = dice_loss_ + wing_Loss_ + cls_loss_ + comm_loss

        scaler.scale(loss).backward()

        # Unscales gradients and calls
        scaler.step(optimizer)
        # Updates the scale for next iteration
        scaler.update()
