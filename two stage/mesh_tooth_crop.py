# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset.

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlpackage          # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""


import os
import time
import sys
from pathlib import Path
from torchvision.ops import nms
import numpy as np
import torch
import trimesh
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import non_max_suppression,xyxy2xywh
from utils.metrics import  box_iou

from utils.torch_utils import select_device, smart_inference_mode
from tooth_utils import read_teeth_mask, mapping_point_clound, FurthestPointSampling, data_normallize, findNearestNeighbors
from tooth_utils import FileErgodic, avaluation_teeth2022, color_space, get_data_to_mesh, CLAS_INDEX, get_rowpoints
from metrics_for_mesh import *
from tooth_number_postprocess import toothNumberingCorrection










def write_obj(vec_points, vec_faces, label, color_space, save_path):
    with open(save_path, "w") as file:
        # 写入顶点和颜色
        for pi in range(vec_points.shape[0]):
            color = color_space[int(label[pi])]
            point = vec_points[pi]
            file.write(f"v {str(point[0])} {str(point[1])} {str(point[2])} {str(color[0])} {str(color[1])} {str(color[2])}\n")

        # 写入面
        vec_faces = vec_faces + 1  # OBJ 文件索引从1开始
        for fi in range(vec_faces.shape[0]):
            face = vec_faces[fi]
            file.write(f"f {str(face[0])} {str(face[1])} {str(face[2])}\n")
def pbox_iou(box1, box2):
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(  # 左上角的点
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2]->[N,1,2]->[N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2]->[1,M,2]->[N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    # 这里之所以要把张量扩增成N*M*2 是因为N个box1和M个box2一共会有N*M个iou值

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # 两个box没有重叠区域
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # (N,)
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # (M,)
    area1 = area1.unsqueeze(1).expand(N, M)  # (N,M)
    area2 = area2.unsqueeze(0).expand(N, M)  # (N,M)

    iou = inter / (area1 + area2 - inter)
    return iou

def test_mapping_point_clound_ellipse(pcoords, conv_coords, row_verts, faces, scalek=10):
    # 1. 扩展矩形框边界 (保持原逻辑)
    label_tid = pcoords[:, 4].copy()
    t_tid = pcoords[:, 5].copy()

    bbox = pcoords[:, :4].copy()
    bbox[:, :2] = bbox[:, :2] - scalek
    bbox[:, 2:4] = bbox[:, 2:4] + scalek

    patch_meshs = []
    teeth_points = []
    teeth_colors = [] # 保持原状
    teeth_indexes = []
    vids = np.array([i for i in range(row_verts.shape[0])])


    tooth_dict = {}
    for i in range(bbox.shape[0]):
        box = bbox[i]
        tid = label_tid[i]
        ttid = t_tid[i]
        # 2. 计算内切椭圆的几何中心和半轴长
        x1, y1, x2, y2 = box
        xc = (x1 + x2) / 2.0  # 中心 X
        yc = (y1 + y2) / 2.0  # 中心 Y
        a = (x2 - x1) / 2.0  # 水平半径
        b = (y2 - y1) / 2.0  # 垂直半径

        # 3. 椭圆筛选逻辑： (x-xc)^2/a^2 + (y-yc)^2/b^2 <= 1
        # 计算所有点相对于中心点的偏移
        dx = conv_coords[:, 0] - xc
        dy = conv_coords[:, 1] - yc

        # 使用向量化计算判定掩码
        # 为了防止 a 或 b 为 0 导致除以零错误，可以加一个极小值 eps
        eps = 1e-6
        mask = (dx ** 2 / (a ** 2 + eps)) + (dy ** 2 / (b ** 2 + eps)) <= 1

        # 4. 根据椭圆掩码提取点和索引
        tpoints = row_verts[mask]
        t_index = vids[mask]
        minz, maxz = np.min(tpoints[:, 2]), np.max(tpoints[:, 2])
        minz = max(maxz-14, minz)
        maskz = tpoints[:, 2] > minz
        tpoints = tpoints[maskz]
        t_index = t_index[maskz]



        label = np.zeros((tpoints.shape[0], 1))


        # new_faces = get_scale_faces(row_verts, faces, t_index)
        #
        # patch_mesh = trimesh.Trimesh(vertices=tpoints, faces=new_faces)
        # patch_meshs.append(patch_mesh)


        teeth_points.append(tpoints)
        tooth_dict[int(tid)] = [ttid, tpoints.astype(np.float32), t_index]

    tooth_dict = {k: tooth_dict[k] for k in sorted(tooth_dict)}

    return teeth_points, tooth_dict
def assign_colors_by_probability(points, probabilities):
    """根据概率值赋予蓝色渐变（浅蓝→深蓝）"""
    probabilities = np.clip(probabilities, 0, 1).flatten()
    colors = np.zeros((len(probabilities), 3))
    colors[:, 2] = np.round(255 * probabilities)  # 蓝色通道随概率增加
    colors[:, 0] = np.round(255 * (1 - probabilities))  # 红色通道递减（可选）
    colors[:, 1] = np.round(255 * (1 - probabilities))  # 绿色通道递减（可选）
    return np.hstack((points, colors.astype(int)))


def save_colored_points(filename, colored_points):
    """
    保存带颜色的点云到txt文件

    参数:
        filename: 输出文件名
        colored_points: Nx6数组，前3列是坐标，后3列是RGB值
    """
    # 确保RGB是整数
    colored_points[:, 3:] = np.round(colored_points[:, 3:])

    # 保存为txt文件，使用空格分隔
    np.savetxt(filename, colored_points, fmt='%.6f %.6f %.6f %d %d %d')
@smart_inference_mode()
def run(
    batch_size=32,  # batch size
    imgsz=512,  # inference size (pixels)
    conf_thres=0.1,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    compute_loss=None,
):
    data= "./data/tooth.yaml"

    file_path = "H:/teethMICCAI2022/raw_data/train/"
    mesh_label_root = "H:/teethMICCAI2022/raw_data/train/"
    img_path ="H:/teethMICCAI2022/correct_data/correct_train/"
    save_root = "./outputs/"

    with open("H:/teethMICCAI2022/private-testing-set.txt", "r", encoding="utf-8") as f:
        data_list = [line.strip() for line in f]

    file_list = []
    FileErgodic(file_path, file_list, ".obj")
    file_list =[file_path for file_path in file_list if os.path.basename(file_path).replace(".obj", "") in data_list]

    model = torch.jit.load('ourbest.torchscript').cuda().eval()

    imgsz = 512
    # Configure
    model.eval()
    model3D = torch.jit.load('rgm_seg.pt')
    model3D.cuda().eval()
    total_time = 0
    TLA, TSA, TIR = [], [], []
    DSCS, ACCS, IOUS = [], [], []
    Toothmask= np.zeros((len(file_list), 9))
    tooth_nums, sam_points = 16, 1024
    file_list_ =["6X2UD6H6_lower", "4W9X0QQI_lower", "28Q7CM3T_upper","07ZWIO50_lower","0LF355FQ_lower", "1FJ3MLSY_upper", "1MWJLE4X_upper", "1RT7WBH4_lower", "2Q8KKZ6V_upper",
                "07ZWIO50_upper","ODC3F7X8_lower", "PY6M5YZ2_lower", "PYAY9ZYX_lower", "T21AWMTR_upper", "UAO672OA_upper"]
    for di in range(0, len(file_list)):
        name_d = os.path.basename(file_list[di]).replace(".obj", "")
        # if name_d not in file_list_:
        #     continue

        #file_list[di] = file_path + file_list[di] +".obj"
        tic = time.time()

        gt_label_dict, pred_label_dict = {}, {}
        print(di, "   ", file_list[di])
        data_name = os.path.split(file_list[di])[-1].replace(".obj", "")
        mesh = trimesh.load(file_list[di])
        vertics = mesh.vertices
        mask_path_ = file_list[di].replace(".obj", ".json")
        gtlabel, gt_instances, thids = read_teeth_mask(mask_path_)

        image, dept_img, conv_coords, row_verts, faces, row_points= get_data_to_mesh(file_list[di], [imgsz, imgsz])
        image = (image / np.max(image)).astype(np.float32)
        image[:, :, 2] = dept_img


        im = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).cuda().float()

        with torch.no_grad():

            nb, _, height, width = im.shape  # batch size, channels, height, width
            # Inference
            preds = model(im)
            preds = non_max_suppression(preds, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=single_cls,max_det=max_det)[0]
            bbox = preds[..., :4].clone()
            cls_score = preds[..., 4].clone()
            nms_index = nms(bbox, cls_score, iou_threshold=0.25)
            preds = preds[nms_index].detach().cpu().numpy()
            pcoords = np.concatenate([preds[..., :4], preds[..., -1:]+1],axis=-1)

            pcoords = toothNumberingCorrection(pcoords)

            Tpoints, sort_tooth_dict = test_mapping_point_clound_ellipse(pcoords, conv_coords, vertics, faces)

            tooth_point_mask = np.zeros((tooth_nums, 2), np.int32)
            data_points = np.zeros((tooth_nums, sam_points, 3), np.float64)

            ou_sort_tooth_dict = {}
            for ki, key in enumerate(sort_tooth_dict):

                input_points = sort_tooth_dict[key][1]
                if input_points.shape[0] > sam_points:
                    se_index = np.random.choice(input_points.shape[0], size=sam_points, replace=False)
                else:
                    se_index = np.random.randint(0, input_points.shape[0], sam_points)
                # se_index = file_data[key]["farthestindex"][2]
                select_points = input_points[se_index]
                tooth_point_mask[int(key) - 1, 0] = 1
                if key < 6 or key > 11:
                    tooth_point_mask[int(key) - 1, 1] = 1
                data_points[int(key) - 1] = select_points
                sort_tooth_dict[key].append(select_points)


            tooth_mask = tooth_point_mask[:, 0]
            # Decentering data
            cpoint = np.mean(data_points[tooth_mask >= 1].reshape(-1, 3), axis=0, keepdims=True)
            data_points[tooth_mask >= 1] = data_points[tooth_mask >= 1] - cpoint.reshape(1, 1, 3)

            data_points = torch.tensor(data_points).unsqueeze(dim=0).cuda().float()
            tooth_point_mask = torch.tensor(tooth_point_mask).unsqueeze(dim=0)

            fl_mask = tooth_point_mask[..., 0].cuda().long()
            match_mask = tooth_point_mask[..., 1].cuda().long()
            with torch.no_grad():
                prelandmarks, matchLandmarks, heat_out, flandmarks, mLandmarks, seg_score = model3D(data_points, fl_mask)


            seg_score = seg_score.squeeze().detach().cpu().numpy()
            index = np.where(tooth_mask>0.5)[0]
            for si in (index):
                sort_tooth_dict[si+1].append(seg_score[si])


            # data_points_numpy = data_points[0, ...].detach().cpu().numpy()
            # seg_score_numpy = seg_score[0, ...].detach().cpu().numpy()
            #
            # for tti in range(data_points_numpy.shape[0]):



            pnums = 1024
            pred_instances = np.zeros((row_verts.shape[0]), np.int32)
            max_probs = np.zeros((row_verts.shape[0]), dtype=np.float32)
            teeth_ids = {}
            for ki, mti in enumerate(sort_tooth_dict):
                ttid = sort_tooth_dict[mti][0]
                t_index = sort_tooth_dict[mti][2]
                input_points = sort_tooth_dict[mti][1]
                points = sort_tooth_dict[mti][3]


                pred3D = sort_tooth_dict[mti][4]
                pred3D = findNearestNeighbors(input_points, points, pred3D)

                colored_points = assign_colors_by_probability(input_points, pred3D)
                save_colored_points("./outputs/heat_points" + str(ki) + ".txt", colored_points)


                # ---------------- 修改开始 ----------------

                # 1. 首先筛选出当前预测概率大于 0.5 的点
                mask_05 = pred3D > 0.5
                valid_t_index = t_index[mask_05]  # 满足 >0.5 条件的全局索引
                valid_pred3D = pred3D[mask_05]  # 对应的预测概率

                # 2. 判断这些点的当前概率是否大于全局已有的最大概率
                # valid_pred3D > max_probs[valid_t_index] 会返回一个布尔数组
                update_mask = valid_pred3D > max_probs[valid_t_index]

                # 3. 提取最终需要更新的全局点索引和对应的概率
                final_update_index = valid_t_index[update_mask]
                final_update_probs = valid_pred3D[update_mask]

                # 4. 执行覆盖：更新实例 ID 和 对应的最大概率记录
                pred_instances[final_update_index] = ttid
                max_probs[final_update_index] = final_update_probs

                # 为了兼容你原代码中对 teeth_ids 的记录逻辑，这里保留原有的 t_index 赋值
                # （注意：这里记录的是所有 >0.5 的点。如果你只希望记录最终真正属于该牙齿的点，
                #  可以将下面这行替换为 t_index = final_update_index）
                t_index = valid_t_index

                # ---------------- 修改结束 ----------------

                teeth_ids[ki] = [ttid, t_index]

            class_weights = torch.ones(9).float()

            htid = np.unique(gtlabel)
            Toothmask[di, htid] = 1
            gt_in = gtlabel.copy()
            dsc = weighting_DSC(torch.tensor(pred_instances), torch.tensor(gt_in), class_weights, class_nums=9)
            iou = weighting_IOU(torch.tensor(pred_instances), torch.tensor(gt_in), class_weights, class_nums=9)
            acc = weighting_ACC(torch.tensor(pred_instances), torch.tensor(gt_in), class_weights, class_nums=9)
            DSCS.append(dsc), IOUS.append(iou), ACCS.append(acc)
            print("dsc = ", dsc.mean(), "  iou = ", iou.mean(), "  acc = ", acc.mean())

            pred_label_dict["thids"] = teeth_ids
            pred_label_dict["instances"] = pred_instances.copy()
            pred_label_dict["mesh_vertices"] = row_verts.copy()
            gt_label_dict["thids"] = thids
            gt_label_dict["instances"] = gtlabel.copy()
            gt_label_dict["mesh_vertices"] = row_verts.copy()

            jaw_TLA, jaw_TIR, jaw_TSA = avaluation_teeth2022(gt_label_dict, pred_label_dict)
            if jaw_TLA < 0.975:
                print("error")
            print("  jaw_TLA=  ", jaw_TLA, "jaw_TIR=  ", jaw_TIR, "jaw_TSA=  ", jaw_TSA)
            TLA.append(jaw_TLA)
            TSA.append(jaw_TSA)
            TIR.append(jaw_TIR)
            toc = time.time()
            total_time = total_time + toc - tic
            print("total time consumed %9f" % (toc - tic) + "s")

            # row_points = get_rowpoints("H:/paper/tooth_segentation/viewer_data/" + data_name + "/" + data_name + ".obj")
            # save_rooth = "H:/paper/tooth_segentation/viewer_data/"
            # # write_obj(row_points, faces, gtlabel, color_space,
            # #           save_rooth + data_name + "/" + data_name + "_" + "gt.obj")
            # write_obj(row_points, faces, pred_instances, color_space,
            #           save_rooth + data_name + "/" + data_name + "_" + "mulsegland.obj")
            # print("")
    print("total_time = ", total_time / len(file_list))
    np.save("TLA.npy", np.array(TLA))
    np.save("TSA.npy", np.array(TSA))
    np.save("TIR.npy", np.array(TIR))
    score = (np.mean(TSA) + np.mean(TLA) + np.mean(TIR)) / 3
    print("TSA : {} +- {}".format(np.mean(TSA), np.std(TSA)))
    print("TLA : {} +- {}".format(np.mean(TLA), np.std(TLA)))
    print("TIR : {} +- {}".format(np.mean(TIR), np.std(TIR)))
    print(" score : ", score)
    ##########################################################
    Toothmask = Toothmask > 0.5
    DSCS, IOUS, ACCS = np.array(DSCS), np.array(IOUS), np.array(ACCS)

    print("DSCS : {} +- {}".format(np.mean(DSCS[Toothmask]), np.std(DSCS[Toothmask])))
    print("IOUS : {} +- {}".format(np.mean(IOUS[Toothmask]), np.std(IOUS[Toothmask])))
    print("ACCS : {} +- {}".format(np.mean(ACCS[Toothmask]), np.std(ACCS[Toothmask])))

    dsc_mean, iou_mean, acc_mean = [], [], []
    for i in range(9):
        dsc = np.mean(DSCS[:, i][Toothmask[:, i]])
        iou = np.mean(IOUS[:, i][Toothmask[:, i]])
        acc = np.mean(ACCS[:, i][Toothmask[:, i]])
        dsc_mean.append(dsc), iou_mean.append(iou), acc_mean.append(acc)

        print(i, "  dsc = ", dsc, "  iou = ", iou, "  acc = ", acc)
    print("mean", "  dsc = ", np.mean(dsc_mean), "  iou = ", np.mean(iou_mean), "  acc = ", np.mean(acc_mean))
    print("over")


if __name__ == "__main__":
    run()
