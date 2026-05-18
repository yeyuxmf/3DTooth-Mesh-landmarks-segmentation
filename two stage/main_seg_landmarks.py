#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data.load_train_data_seg_land import TrainData
from data.load_test_data_seg_land import TestData
from torch.utils.data import DataLoader
from model.tooth_seg_landmark_net import ToothLandmark
from model.loss import dice_loss
import config.config as cfg
from util import IOStream, accumulate_net, cal_acc_seg_land
from model.regression_loss import focal_loss

def model_initial(model, model_name):
    # 加载预训练模型
    pretrained_dict = torch.load(model_name)["model"]
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    # pretrained_dictf = {k.replace('module.', ""): v for k, v in pretrained_dict.items() if k.replace('module.', "") in model_dict}
    pretrained_dictf = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dictf)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    print("over")

def add_gaussion_noise(Rteeth_points):
    # add  noise
    scalek = np.random.randint(0, 150, 1)[0] / 1000.0
    gaussson_noise = torch.normal(0, 1, Rteeth_points.shape)
    gaussson_noise = gaussson_noise /torch.max(gaussson_noise)
    train_data = Rteeth_points + gaussson_noise.cuda().float()*scalek

    # Rteeth_points_numpy = Rteeth_points.cpu().numpy()[0]
    # gaussson_noise_numpy = train_data.cpu().numpy()[0]
    #
    # file_ = open("./outputs/tttt"  + ".txt", "w")
    # for tid in range(Rteeth_points_numpy.shape[0]):
    #     points = Rteeth_points_numpy[tid]
    #     for i in range(points.shape[0]):
    #         file_.write(str(points[i][0]) + " " + str(points[i][1]) + " " + str(points[i][2]) + "\n")
    #
    # file_.close()
    #
    # file_ = open("./outputs/ggggg.txt", "w")
    # for tid in range(Rteeth_points_numpy.shape[0]):
    #     points = gaussson_noise_numpy[tid]
    #     for i in range(points.shape[0]):
    #         file_.write(str(points[i][0]) + " " + str(points[i][1]) + " " + str(points[i][2]) + "\n")
    #
    # file_.close()


    return train_data

def calculate_segmentation_metrics(y_true, y_pred, num_classes):
    """
    计算分割指标: Accuracy, mIoU, DSC
    :param y_true: 真实标签 (N,)
    :param y_pred: 预测标签 (N,)
    [cite_start]:param num_classes: 类别总数 (论文中提到为 17 类，包含 16 颗牙齿和牙龈 )
    """
    # 1. 计算 Accuracy (Acc)
    # [cite_start]逻辑：预测正确的点数 / 总点数
    acc = np.mean(y_true == y_pred)

    # 初始化用于存储每类 IoU 和 DSC 的列表
    ious = []
    dices = []

    for i in range(num_classes):
        # 获取第 i 类的布尔掩码
        true_mask = (y_true == i)
        pred_mask = (y_pred == i)

        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        target_sum = true_mask.sum()
        pred_sum = pred_mask.sum()

        # 2. 计算 IoU (Intersection over Union)
        # [cite_start]逻辑：交集 / 并集
        if union == 0:
            iou = 1.0  # 如果该类在真值和预测中都不存在，视为满分
        else:
            iou = intersection / union
        ious.append(iou)

        # 3. 计算 DSC (Dice Similarity Coefficient)
        # [cite_start]逻辑：2 * 交集 / (预测数 + 真值数)
        if (target_sum + pred_sum) == 0:
            dice = 1.0
        else:
            dice = (2. * intersection) / (target_sum + pred_sum)
        dices.append(dice)

    # 计算均值 (mIoU)
    miou = np.mean(ious)
    # 计算均值 (mDSC)
    mdsc = np.mean(dices)

    return acc, miou, mdsc

def add_gaussian_noise(landmarks, sigma=1.0, max_offset=1.0):
    """
    landmarks: 原始地标坐标，形状为 (N, 2) 或 (2,)
    sigma: 高斯噪声的标准差
    max_offset: 最大允许偏移量（绝对值）
    """
    if not isinstance(landmarks, torch.Tensor):
        landmarks = torch.tensor(landmarks, dtype=torch.float32)

    # 生成高斯噪声并截断
    noise = torch.randn_like(landmarks) * sigma  # 标准正态分布 * sigma
    noise = torch.clamp(noise, -max_offset, max_offset)  # 限制范围

    noisy_landmarks = landmarks + noise
    return noisy_landmarks

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('./outputs/' + args.exp_name):
        os.makedirs('./outputs/' + args.exp_name)
    if not os.path.exists('./outputs/' + args.exp_name + '/' + 'models'):
        os.makedirs('./outputs/' + args.exp_name + '/' + 'models')
    os.system('cp main_cls.py outputs' + '/' + args.exp_name + '/' + 'main_cls.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_path = "E:/DataSet/mesh_tooth_landmark_data/seg_landmarks_train_data/train/"
    train_loader = DataLoader(TrainData(file_path), num_workers=0,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    file_path = "E:/DataSet/mesh_tooth_landmark_data/seg_landmarks_train_data/test/"
    test_loader = DataLoader(TestData(file_path), num_workers=0,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    model = ToothLandmark()#
    # model_ema = model
    model_path = "./outputs/seg_land_model_best.pth"
    # model_initial(model, model_path)


    # input_data1 = torch.rand((1, 16, 512, 3)).cuda().float()
    # input_data2 = torch.ones((1, 16)).cuda().bool()
    # scripted_module = torch.jit.trace(model.cuda().eval(), (input_data1, input_data2))
    # torch.jit.save(scripted_module, './save_model/tooth_seg_landmarks.pt')
    # model = torch.jit.load('./save_model/tooth_seg_landmarks3.pt').cuda().eval().float()
    #


    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:

        print("Use SGD")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6, last_epoch = -1)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=20, gamma=0.7)
    model.cuda()

    scaler = GradScaler()
    inter_nums = len(train_loader)


    max_ap_ar = 0
    max_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        total_loss = 0
        seg_loss = 0
        f_coord_loss = 0
        f_cls_loss = 0
        m_coord_loss = 0
        m_cls_loss = 0
        diceloss = 0
        fcoord_loss =0
        mcoord_loss =0

        fdloss = 0
        mdloss = 0

        noise_floss = 0
        noise_mloss = 0
        noise_diceloss = 0

        model.train()
        scheduler.step()
        tic = time.time()
        nums =0
        land_nums = 0
        for batch_data, data_label, tooth_mask, land_mask, label_landmarks, heat_map in train_loader:
            #label_landmarks = add_gaussian_noise(label_landmarks)

            nums = nums +1
            batch_data = batch_data.cuda().float()
            data_label = data_label.cuda().float()
            heat_map = heat_map.cuda().float()
            land_mask = land_mask.cuda().long()
            label_landmarks = label_landmarks.cuda().float()
            mask = torch.sum(land_mask, dim=-1)

            fl_mask = land_mask[..., :5].bool()
            match_mask = land_mask[..., 5:].bool()
            fl_landmarks = label_landmarks[..., :5, :]
            ml_landmarks = label_landmarks[..., 5:, :]

            optimizer.zero_grad()
            with autocast():
                #batch_data = add_gaussion_noise(batch_data)
                prelandmarks, matchLandmarks, heat_out, flandmarks, mLandmarks, seg_score = model(batch_data, mask)

            seg_loss_ = focal_loss(seg_score, data_label)

            if torch.sum(land_mask)>=1:
                land_nums = land_nums +1
                diceloss_ = dice_loss(heat_out[0:1, ...][mask>=1], heat_map[mask>=1])
                f_coord_loss_, f_cls_loss_, fcoord_loss_, floss = model.fixed_loss(prelandmarks[0:1, ...], fl_landmarks, fl_mask)
                fd_coord_loss_, fd_cls_loss_, fdcoord_loss_, fdloss_ = model.fixed_loss(flandmarks, fl_landmarks, fl_mask)


                if torch.sum(match_mask) > 0:
                    m_coord_loss_,  m_cls_loss_, mcoord_loss_, mloss = model.match_loss(matchLandmarks[0:1, ...], ml_landmarks, match_mask)
                    md_coord_loss_,  md_cls_loss_, mdcoord_loss_, mdloss_ = model.match_loss(mLandmarks, ml_landmarks, match_mask)

                    loss = (floss + mloss + diceloss_) + (mdloss_ + fdloss_) + seg_loss_
                else:
                    loss = (floss + diceloss_)+ (fdloss_) + seg_loss_
            else:
                loss = seg_loss_

                # prelandmarks_numpy = prelandmarks.squeeze().detach().cpu().numpy()
            # label_landmarks_numpy = label_landmarks.squeeze().detach().cpu().numpy()

            scaler.scale(loss).backward()

            # Unscales gradients and calls
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()

            # if model_ema is not None and epoch % 1 == 0:
            #     accumulate_net(model_ema, model, 0.5 ** (8 / 10000.0))


            total_loss += loss.item()
            seg_loss += seg_loss_.item()
            if torch.sum(land_mask) >= 1:
                f_coord_loss += f_coord_loss_.item()
                f_cls_loss += f_cls_loss_.item()
                m_coord_loss += m_coord_loss_.item()
                m_cls_loss += m_cls_loss_.item()

                fcoord_loss += fcoord_loss_.item()
                mcoord_loss += mcoord_loss_.item()
                diceloss += diceloss_.item()

                fdloss += fdloss_.item()
                mdloss += mdloss_.item()

                noise_floss += fcoord_loss_.item()
                noise_mloss += fcoord_loss_.item()
                noise_diceloss += fcoord_loss_.item()


            if nums % cfg.LOSSNUMS == 0:
                toc = time.time()
                total_loss = total_loss / cfg.LOSSNUMS
                seg_loss = seg_loss / cfg.LOSSNUMS
                if land_nums >= 1:
                    f_coord_loss = f_coord_loss / land_nums
                    f_cls_loss = f_cls_loss / land_nums
                    m_coord_loss = m_coord_loss / land_nums
                    m_cls_loss = m_cls_loss / land_nums
                    fcoord_loss = fcoord_loss / land_nums
                    mcoord_loss = mcoord_loss / land_nums
                    diceloss = diceloss / land_nums

                    fdloss = fdloss / land_nums
                    mdloss = mdloss /land_nums

                    noise_floss = noise_floss / land_nums
                    noise_mloss = noise_mloss / land_nums
                    noise_diceloss = noise_diceloss / land_nums


                    print("lr = ", optimizer.param_groups[0]['lr'], "fcoord_loss = ", fcoord_loss, "mcoord_loss = ", mcoord_loss, "diceloss = ", diceloss)
                    print("fdloss = ", fdloss, "mdloss = ", mdloss)

                print(
                    'epoch %d /%d,epoch %d /%d, total_loss: %.6f, seg_loss: %.6f, f_coord_loss: %.6f, f_cls_loss: %.6f, m_coord_loss: %.6f, m_cls_loss: %.6f, const time: %.6f' % (
                        epoch, args.epochs, nums, inter_nums, total_loss, seg_loss, f_coord_loss, f_cls_loss, m_coord_loss, m_cls_loss, toc - tic))
                land_nums = 0
                total_loss= 0
                seg_loss = 0
                f_coord_loss = 0
                f_cls_loss = 0
                m_coord_loss = 0
                m_cls_loss = 0

                fcoord_loss = 0
                mcoord_loss = 0
                diceloss = 0

                fdloss = 0
                mdloss = 0

                noise_floss = 0
                noise_mloss = 0
                noise_diceloss = 0

                tic = time.time()

        if epoch+1 >=0 and 0 == (epoch+1)%1:
            model.eval()
            #
            fm_cdv, all_metrics, score, fm_cdvd, all_metricsd, scoresd = cal_acc_seg_land(model, test_loader)


            ap_v = np.mean(np.array(list(all_metrics["AP"].values())))
            ar_v = np.mean(np.array(list(all_metrics["AR"].values())))
            ap_ar_v = (ap_v + ar_v) /2.0
            if ap_ar_v > max_ap_ar:
                max_ap_ar = ap_ar_v
                torch.save({'model': model.state_dict(), 'epoch': epoch}, 'outputs/' + 'seg_land_model_best.pth')
            print("max_ap_ar = ", max_ap_ar)
            print("cur_ap_ar_v = ", ap_ar_v)
            print("fm_cdv = ", fm_cdv)

            print(all_metrics["AP"])
            print(all_metrics["AR"])
            # print("***********************************************************\n")
            print(all_metricsd["AP"])
            print(all_metricsd["AR"])
            if max_acc < np.mean(np.array(list(all_metricsd["AP"].values()))):
                max_acc = np.mean(np.array(list(all_metricsd["AP"].values())))
            print("max_acc = ", max_acc)
            print("meanAP = ", np.mean(np.array(list(all_metricsd["AP"].values()))))

    torch.save({'model': model.state_dict(), 'epoch': epoch}, 'outputs/seg_land_model_final' + "" + '.pth')





if __name__ == "__main__":


    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='cls_1024', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=101, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')#1.2*1e-4
    parser.add_argument('--lr', type=float, default=1.2*1e-4, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=2048, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)

