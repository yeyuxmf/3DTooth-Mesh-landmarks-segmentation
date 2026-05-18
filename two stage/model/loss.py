import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from model.realnvp import RealNVP
from model.regression_loss import RLELoss, WingLoss,focal_loss

class matchLandmarkLoss(nn.Module):
    def __init__(self, scale=30.0, in_channels=3):
        super(matchLandmarkLoss, self).__init__()
        self.scale = scale  # 建议设为牙齿区域的半径，如 30.0
        self.wingloss = WingLoss()

        self.match_loss = RLELoss(use_target_weight=False,
                            size_average=True,
                            residual=True,
                            q_dis='laplace',
                            in_channels=3)
        self.log_vars = nn.Parameter(torch.zeros(2))
    def forward(self, output, target, ma_mask):
        """
        output: (B*N, 5, 7) -> [x, y, z, log_sigma, exists_logit]
        target: (B*N, 5, 3)
        mask:   (B*N, 5)    -> 1存在, 0不存在
        """

        mask = torch.sum(ma_mask, dim=-1)
        mask = torch.gt(mask,  1)


        output = output[mask].float()
        target = target[mask]
        tooth_mask = ma_mask[mask]

        TN = target.shape[0]
        output = output.reshape(TN, -1, 7)

        # 提取各个分支
        pred_coords = output[..., :3]
        # 修改点1: 增加 0.1 偏移，初始 sigma 约为 0.6，防止训练前期分母爆炸
        pred_sigmas = output[..., 3:6]
        exists_logits = output[..., 6]

        total_rle_loss = 0
        total_cls_loss = 0
        m_preds, m_sigmas, m_gts = [], [], []

        for i in range(TN):
            m = torch.nonzero(tooth_mask[i] > 0).squeeze()
            num_gt = tooth_mask[i].sum().item()
            gt_points = target[i, m].reshape(-1, 3)

            with torch.no_grad():
                # 修改点2: 距离代价除以尺度，使其与分类代价量级匹配
                dist_cost = torch.cdist(pred_coords[i], gt_points, p=2) / self.scale
                cls_cost = -torch.sigmoid(exists_logits[i]).unsqueeze(1).repeat(1, num_gt)
                cost = dist_cost + 0.5 * cls_cost
                pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())

            # 分类损失
            cls_target = torch.zeros(output.shape[1], device=output.device)
            cls_target[pred_idx] = 1.0
            total_cls_loss += focal_loss(exists_logits[i].sigmoid(), cls_target) # F.binary_cross_entropy_with_logits(exists_logits[i], cls_target)

            m_preds.append(pred_coords[i][pred_idx])
            m_gts.append(gt_points[gt_idx])
            m_sigmas.append(pred_sigmas[i][pred_idx])
        total_cls_loss = total_cls_loss / TN

        # 拼接匹配后的数据
        flat_pred = torch.cat(m_preds, dim=0)
        flat_target = torch.cat(m_gts, dim=0)
        flat_sigma = torch.cat(m_sigmas, dim=0)

        l1_coord_loss = F.smooth_l1_loss(flat_pred, flat_target)

        diff_pred = flat_pred.unsqueeze(2) - flat_pred.unsqueeze(1)
        diff_target = flat_target.unsqueeze(2) - flat_target.unsqueeze(1)
        rela_coord_loss = self.wingloss(diff_pred, diff_target) / diff_pred.shape[0]


        # 修改点3: 统一尺度计算 Error。不再执行 (x+50)/100
        # 这样 error = (实际物理距离偏差) / (sigma * 30mm)
        total_rle_loss = self.wingloss(flat_pred, flat_target)#self.match_loss(torch.cat([flat_pred, flat_sigma], dim=-1).unsqueeze(dim=0), flat_target.unsqueeze(dim=0))


        precision_coord = torch.exp(-self.log_vars[0])
        loss_1 = precision_coord * total_rle_loss + self.log_vars[0]

        precision_cls = torch.exp(-self.log_vars[1])
        loss_2 = precision_cls * total_cls_loss + self.log_vars[1]

        mloss = loss_1 + loss_2

        mloss = total_rle_loss + total_cls_loss + rela_coord_loss

        return total_rle_loss, total_cls_loss, l1_coord_loss, mloss

class fixedLandmarkLoss(nn.Module):
    """
    专门用于牙齿 5 类固定关键点 (Mesial, Distal, Inner, Outer, Facial) 的 RLE 损失函数。
    适用于坐标范围在 [-50, 50] 左右的 3D 数据。
    """

    def __init__(self, scale=1.0, in_channels=3):
        super(fixedLandmarkLoss, self).__init__()
        self.scale = scale  # 坐标缩放因子，将 [-50, 50] 映射到 [-1, 1] 附近
        # 初始化流模型，in_channels=3 对应 (x, y, z) 的残差学习
        self.wingloss = WingLoss()
        self.fixed_loss = RLELoss(use_target_weight=False,
                            size_average=True,
                            residual=True,
                            q_dis='laplace',
                            in_channels=3)

        self.log_vars = nn.Parameter(torch.zeros(2))
    def forward(self, prelandmarks, fl_landmarks, fl_mask):
        """
        Args:
            output: (BN, 5, 6) -> 5个点的预测，每个点包含 [x, y, z, log_sigma_x, log_sigma_y, log_sigma_z]
            target: (BN, 5, 3) -> 5个点的真实 3D 坐标
        """

        mask = torch.sum(fl_mask, dim=-1)

        mask = torch.gt(mask,  1)

        B, Tv = mask.shape
        pcoords = prelandmarks.squeeze().reshape(B, Tv, 5, 7)[..., :3]
        Mesial = torch.cat([pcoords[:7,0, :], pcoords[8:-1,1, :], pcoords[7:8,0, :]],dim=0)
        Distal = torch.cat([pcoords[1:8, 1, :], pcoords[9:,0, :], pcoords[8:9,0, :]],dim=0)


        prelandmarks = prelandmarks[mask]
        gtlandmarks = fl_landmarks[mask]
        tooth_mask = fl_mask[mask]

        TN = gtlandmarks.shape[0]
        prelandmarks = prelandmarks.reshape(TN, -1, 7)

        all_sigmas = prelandmarks[..., 3:6]
        pred = prelandmarks[..., :3][tooth_mask]
        sigma = all_sigmas[tooth_mask]
        gt = gtlandmarks[tooth_mask]  # 已经是提取好的

        l1_coord_loss = F.smooth_l1_loss(pred, gt)

        # 尺度缩放计算
        coord_loss = self.wingloss(pred, gt)#self.fixed_loss(torch.cat([pred, sigma], dim=-1).unsqueeze(dim=0), gt.unsqueeze(dim=0))

        diff_pred = pred.unsqueeze(2) - pred.unsqueeze(1)
        diff_target = gt.unsqueeze(2) - gt.unsqueeze(1)
        rela_coord_loss = self.wingloss(diff_pred, diff_target) / pred.shape[0]

        # 分类损失 (BCE)
        precls = prelandmarks[..., 6]
        cls_target = torch.zeros_like(precls)
        cls_target[tooth_mask] = 1.0
        cls_loss = focal_loss(precls.sigmoid(), cls_target)#F.binary_cross_entropy_with_logits(precls, cls_target)

        precision_coord = torch.exp(-self.log_vars[0])
        loss_1 = precision_coord * coord_loss + self.log_vars[0]

        precision_cls = torch.exp(-self.log_vars[1])
        loss_2 = precision_cls * cls_loss + self.log_vars[1]

        floss = coord_loss +  cls_loss + rela_coord_loss#loss_1 + loss_2

        return coord_loss, cls_loss, l1_coord_loss, floss


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def normal_consistency_loss(pred_normals, gt_normals, mask=None):
    """
    计算法线一致性损失（余弦距离）
    pred_normals/gt_normals: [B, TN, K, 3]
    mask: [B, TN, K] 牙齿存在且关键点有效的掩码
    """
    # 归一化为单位向量
    p_n = F.normalize(pred_normals, p=2, dim=-1)
    g_n = F.normalize(gt_normals, p=2, dim=-1)

    # 计算余弦相似度并转为距离 (0 到 2)
    cosine_sim = torch.sum(p_n * g_n, dim=-1)
    loss = 1.0 - cosine_sim

    if mask is not None:
        return (loss * mask).sum() / (mask.sum() + 1e-6)
    return loss.mean()


def curvature_regression_loss(pred_curv, gt_curv, mask=None, use_smooth_l1=True):
    """
    计算曲率回归损失
    pred_curv/gt_curv: [B, TN, K] 或 [B, TN, K, 1]
    """
    if pred_curv.shape != gt_curv.shape:
        pred_curv = pred_curv.view_as(gt_curv)

    if use_smooth_l1:
        loss = F.smooth_l1_loss(pred_curv, gt_curv, reduction='none')
    else:
        loss = F.mse_loss(pred_curv, gt_curv, reduction='none')

    if mask is not None:
        return (loss * mask).sum() / (mask.sum() + 1e-6)
    return loss.mean()