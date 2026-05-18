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
from data.load_train_data_cls import TrainData
from data.load_test_data import TestData
from torch.utils.data import DataLoader
from model.tooth_landmarks_onetAI  import ToothLandmark
from model.loss import diceLoss, focalLoss, WingLoss, ToothHeatmapLoss, ToothOffsetLoss
import config.config as cfg
from util import IOStream, accumulate_net, cal_acc_seg_land_cls


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

    file_path = "G:/teethMICCAI2022/train_land_onenet_data/train/"
    train_loader = DataLoader(TrainData(file_path), num_workers=0,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    file_path = "G:/teethMICCAI2022/train_land_onenet_data/test/"
    test_loader = DataLoader(TestData(file_path), num_workers=0,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    model = ToothLandmark()#
    # model_ema = model
    model_path = "./save_model/seg_land_model_cls_best.pth"
    model_initial(model, model_path)


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
    wingLoss = WingLoss()
    heatt_loss = ToothHeatmapLoss()
    off_loss = ToothOffsetLoss()
    max_ap_ar = 0
    max_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        total_loss = 0
        dice_loss = 0
        wing_Loss =0
        cls_loss = 0

        model.train()
        scheduler.step()
        tic = time.time()
        nums =0
        land_nums = 0
        # for batch_data, heat_map, offest_map, gtcls, mask in train_loader:
        #     #label_landmarks = add_gaussian_noise(label_landmarks)
        #
        #     nums = nums +1
        #     batch_data = batch_data.cuda().float()
        #     heat_map = heat_map.cuda().float()
        #     offest_map = offest_map.cuda().float()
        #     gtcls = gtcls.cuda().float()
        #     mask = mask.cuda().bool()
        #
        #     optimizer.zero_grad()
        #     with autocast():
        #         #batch_data = add_gaussion_noise(batch_data)
        #         preheat_map, preoff_map, cls = model(batch_data)
        #     dice_loss_ = heatt_loss(preheat_map, heat_map, mask)
        #     cls_loss_ = heatt_loss(cls, gtcls, gtcls.cuda().bool())
        #     #seg_loss_ = focalLoss(preheat_map, heat_map)
        #     #wing_Loss_ = wingLoss(preoff_map[mask], offest_map[mask])
        #     wing_Loss_ = off_loss(preoff_map.float(), offest_map.float(), mask)
        #     loss = dice_loss_ + wing_Loss_ + cls_loss_
        #
        #
        #     scaler.scale(loss).backward()
        #
        #     # Unscales gradients and calls
        #     scaler.step(optimizer)
        #     # Updates the scale for next iteration
        #     scaler.update()
        #
        #     # if model_ema is not None and epoch % 1 == 0:
        #     #     accumulate_net(model_ema, model, 0.5 ** (8 / 10000.0))
        #
        #
        #     total_loss += loss.item()
        #     dice_loss += dice_loss_.item()
        #     wing_Loss += wing_Loss_.item()
        #     cls_loss += cls_loss_.item()
        #
        #     if nums % cfg.LOSSNUMS == 0:
        #         toc = time.time()
        #         total_loss = total_loss / cfg.LOSSNUMS
        #         dice_loss = dice_loss / cfg.LOSSNUMS
        #         wing_Loss = wing_Loss / cfg.LOSSNUMS
        #         cls_loss = cls_loss / cfg.LOSSNUMS
        #
        #         print("lr = ", optimizer.param_groups[0]['lr'])
        #
        #
        #         print(
        #             'epoch %d /%d,epoch %d /%d, total_loss: %.6f, dice_loss: %.6f, wing_Loss: %.6f, cls_loss: %.6f, const time: %.6f' % (
        #                 epoch, args.epochs, nums, inter_nums, total_loss, dice_loss, wing_Loss, cls_loss, toc - tic))
        #         total_loss = 0
        #         dice_loss = 0
        #         wing_Loss = 0
        #         cls_loss = 0
        #         tic = time.time()

        if epoch+1 >=90 and 0 == (epoch+1)%1:
            model.eval()
            #
            fm_cdv, all_metrics, score = cal_acc_seg_land_cls(model, test_loader)

            # print("***********************************************************\n")
            print(all_metrics["AP"])
            print(all_metrics["AR"])
            if max_acc < np.mean(np.array(list(all_metrics["AP"].values()))):
                max_acc = np.mean(np.array(list(all_metrics["AP"].values())))
                # torch.save({'model': model.state_dict(), 'epoch': epoch}, 'outputs/' + 'seg_land_model_cls_best.pth')
            print("max_acc = ", max_acc)
            print("meanAP = ", np.mean(np.array(list(all_metrics["AP"].values()))))

    torch.save({'model': model.state_dict(), 'epoch': epoch}, 'outputs/seg_land_model_cls_final' + "" + '.pth')





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
                        help='Use SGD')
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

