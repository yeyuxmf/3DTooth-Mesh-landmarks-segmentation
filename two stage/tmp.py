import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from model.realnvp import RealNVP

class matchLandmarkLoss(nn.Module):
    def __init__(self, scale=50.0, in_channels=3):
        super(matchLandmarkLoss, self).__init__()
        self.scale = scale
        # 这里的 RealNVP 依然处理 3D 坐标残差
        self.flow_model = RealNVP(in_channels=in_channels)

    def forward(self, output, target, mask):
        """
        output: (B*N, 5, 7) -> [x, y, z, log_sigma, exists_logit]
        target: (B*N, 5, 3)
        mask:   (B*N, 5)    -> 1存在, 0不存在
        """
        B, K, C = mask.unsqueeze(dim=0).shape

        mask = mask.squeeze()
        tooth_mask = torch.sum(mask, dim=-1)

        indices = torch.nonzero(tooth_mask.squeeze() > 0).reshape(-1)
        output = torch.cat([output[ind] for ind in indices], dim=0).reshape(indices.shape[0], C, -1).float()
        target = torch.stack([target[ind] for ind in indices], dim=0).float()
        mask = torch.stack([mask[ind] for ind in indices], dim=0)


        # 提取各个分支
        pred_coords = output[:, :, :3]
        pred_sigmas = output[:, :, 3:6].sigmoid()
        exists_logits = output[:, :, 6]  # (BN, 6)

        total_rle_loss = 0
        total_cls_loss = 0
        valid_teeth = 0
        BN = indices.shape[0]
        for i in range(BN):
            m = torch.nonzero(mask[i] > 0).squeeze()
            num_gt = mask[i].sum().item()

            # 1. 准备当前牙齿的 GT
            gt_points = target[i, m].reshape(-1, 3)  # (num_gt, 3)

            # 2. 匈牙利匹配 (Hungarian Matching)
            # 计算所有预测点与真实点之间的 Cost Matrix
            # 参考 Poseur，代价通常包含：距离代价 + 分类代价
            with torch.no_grad():
                dist_cost = torch.cdist(pred_coords[i] / self.scale, gt_points / self.scale, p=2)  # (6, num_gt)
                # 匹配时考虑分类预测，让擅长预测该点的 Query 去匹配
                cls_cost = -exists_logits[i].unsqueeze(1).repeat(1, num_gt)
                cost = dist_cost + 0.5 * cls_cost

                # 寻找最佳匹配索引
                pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())

            # 3. 计算分类损失 (BCE)
            # 创建分类标签：匹配到的点为1，没匹配到的为0
            cls_target = torch.zeros(C, device=output.device)
            cls_target[pred_idx] = 1.0
            total_cls_loss += F.binary_cross_entropy_with_logits(exists_logits[i], cls_target)



            m_pred = pred_coords[i][pred_idx]
            m_gt = gt_points[gt_idx]

            total_rle_loss += F.smooth_l1_loss(m_pred, m_gt)

            # 4. 计算回归损失 (RLE) - 仅针对匹配上的正样本
        #     if num_gt > 0:
        #         m_pred = pred_coords[i][pred_idx]
        #         m_gt = gt_points[gt_idx]
        #         m_sigma = pred_sigmas[i][pred_idx]
        #
        #         error = (m_pred / self.scale - m_gt / self.scale) / (m_sigma + 1e-9)
        #
        #         # RLE 核心计算
        #         log_phi = self.flow_model.log_prob(error)
        #         log_sigma = torch.log(m_sigma)
        #         loss_q = torch.log(m_sigma * 2) + torch.abs(error)
        #         nf_loss = log_sigma - log_phi.view(-1, 1)
        #
        #         total_rle_loss += (nf_loss + loss_q).mean()
        #         valid_teeth += 1
        #
        return total_rle_loss/BN, total_cls_loss/BN #(total_rle_loss / (valid_teeth + 1e-6)) + total_cls_loss / BN


class fixedLandmarkLoss(nn.Module):
    """
    专门用于牙齿 5 类固定关键点 (Mesial, Distal, Inner, Outer, Facial) 的 RLE 损失函数。
    适用于坐标范围在 [-50, 50] 左右的 3D 数据。
    """

    def __init__(self, scale=50.0, in_channels=3):
        super(fixedLandmarkLoss, self).__init__()
        self.scale = scale  # 坐标缩放因子，将 [-50, 50] 映射到 [-1, 1] 附近
        # 初始化流模型，in_channels=3 对应 (x, y, z) 的残差学习
        self.flow_model = RealNVP(in_channels=in_channels)

    def forward(self, prelandmarks, fl_landmarks, fl_mask):
        """
        Args:
            output: (BN, 5, 6) -> 5个点的预测，每个点包含 [x, y, z, log_sigma_x, log_sigma_y, log_sigma_z]
            target: (BN, 5, 3) -> 5个点的真实 3D 坐标
        """

        mask = torch.sum(fl_mask, dim=-1)

        indices = torch.nonzero(mask.squeeze() > 0).squeeze()

        prelandmarks = [prelandmarks[ind].reshape(-1, 4) for ind in indices]
        tooth_mask = [fl_mask[ind] for ind in indices]

        prelandmarks = torch.cat(prelandmarks, dim=0)
        tooth_mask = torch.cat(tooth_mask, dim=0)

        precls = prelandmarks[..., 3]
        prelandmarks = prelandmarks[..., :3][tooth_mask]


        gtlandmarks = [fl_landmarks[ind].reshape(-1, 3)[fl_mask[ind]] for ind in indices]
        gtlandmarks = torch.cat(gtlandmarks, dim=0)


        coord_loss = F.smooth_l1_loss(prelandmarks, gtlandmarks)


        cls_target = torch.zeros(precls.shape[0], device=precls.device)
        cls_target[tooth_mask] = 1.0
        cls_loss = F.binary_cross_entropy_with_logits(precls, cls_target)

        #print("")






        # BN, K, _ = output.shape
        #
        # # 1. 拆分预测的坐标和不确定性 (sigma)
        # pred = output[:, :, :3]
        # # 使用 sigmoid 确保标准差 sigma 为正数
        # sigma = output[:, :, 3:6].sigmoid()
        #
        # # 2. 展平数据以便进行批处理计算
        # # (BN * 5, 3)
        # flat_pred = pred.reshape(-1, 3)
        # flat_target = target.reshape(-1, 3)
        # flat_sigma = sigma.reshape(-1, 3)
        #
        # # 3. 计算归一化后的标准误差 (Standardized Error)
        # # 将原始坐标除以 scale，使误差分布在 Flow Model 易于处理的范围内
        # error = (flat_pred / self.scale - flat_target / self.scale) / (flat_sigma + 1e-9)
        #
        # # 4. 计算 Flow Model 的对数似然 (Log-Likelihood)
        # # log_phi 代表误差在习得分布下的概率
        # log_phi = self.flow_model.log_prob(error)  # 输出维度: (BN * 5,)
        #
        # # 5. 计算基础分布损失 (这里假设基础分布为 Laplace 分布)
        # # Loss = log(sigma) - log_prob(error)
        # log_sigma = torch.log(flat_sigma)  # (BN * 5, 3)
        #
        # # 基础拉普拉斯负似然项
        # loss_q = torch.log(flat_sigma * 2) + torch.abs(error)
        #
        # # 结合 RealNVP 学习到的残差修正
        # nf_loss = log_sigma - log_phi.view(-1, 1)
        #
        # # 6. 综合损失
        # loss = nf_loss + loss_q

        return coord_loss, cls_loss #loss.mean()