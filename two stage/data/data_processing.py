import os
import time
import vtk
import json
import cv2
import trimesh
from torchvision.ops import nms
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from utils.general import non_max_suppression
from tooth_utils import read_teeth_mask, mapping_point_clound_ellipse, test_mapping_point_clound_ellipse
from tooth_utils import FileErgodic, get_data_to_mesh
from signal_tooth_split import view_row_key_point
from data.tooth_number_postprocess import toothNumberingCorrection
color_dict = {
        0: [255, 0, 0],     # 鲜红
        8: [0, 255, 0],     # 翠绿
        7: [0, 0, 255],     # 纯蓝
        6: [255, 255, 0],   # 亮黄
        5: [255, 0, 255],   # 品红
        4: [0, 255, 255],   # 青色
        3: [255, 165, 0],   # 橙色
        2: [0, 32, 240],    # 紫色 (修正为更亮的紫)
        1: [255, 12, 0],  # 紫色 (修正为更亮的紫)

    9: [0, 255, 0],  # 翠绿
    10: [0, 0, 255],  # 纯蓝
    11: [255, 255, 0],  # 亮黄
    12: [255, 0, 255],  # 品红
    13: [0, 255, 255],  # 青色
    14: [255, 165, 0],  # 橙色
    15: [160, 32, 240],  # 紫色 (修正为更亮的紫)
    16: [255, 12, 0]  # 紫色 (修正为更亮的紫)
    }


def write_data(data_path, label, color_space, save_dapth):

    reader = vtk.vtkOBJReader()
    reader.SetFileName(data_path)
    reader.Update()
    polydata = reader.GetOutput()

    # 创建颜色数组
    num_points = polydata.GetNumberOfPoints()
    color_array = vtk.vtkUnsignedCharArray()
    color_array.SetNumberOfComponents(3)  # RGB
    color_array.SetName("Colors")  # 设置颜色数组的名称

    # 将颜色值添加到颜色数组中
    for i in range(num_points):
        colors = color_space[int(label[i])]
        r, g, b = colors[i]  # 获取每个顶点的 RGB 值
        color_array.InsertNextTuple3(r, g, b)

    # 将颜色数组添加到 polydata 中
    polydata.GetPointData().SetScalars(color_array)

    # 保存为 OBJ 文件
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(save_dapth)
    writer.SetInputData(polydata)
    writer.Write()


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


def read_landmarks(file_path):

    gtkey_points = {'Mesial': [], 'Distal': [], 'InnerPoint': [], 'OuterPoint': [], 'FacialPoint': [], 'Cusp': []}
    annots = json.load(open(file_path))
    landmarks = annots['objects']
    for i, kp in enumerate(landmarks):
        gtkey_points[kp["class"]].append(kp["coord"])

    return gtkey_points


def compute_distance_matrix_numpy(point_clouds, landmarks):
    """
    point_clouds: (N, P, 3)
    landmarks: (M, 3)
    return: (N, M)
    """
    # 扩展维度以进行广播
    # (N, P, 1, 3) - (1, 1, M, 3) -> (N, P, M, 3)
    # 通过这种方式，每个点云的每个点都与每个地标相减
    diff = point_clouds[:, :, np.newaxis, :] - landmarks[np.newaxis, np.newaxis, :, :]

    # 计算欧式距离的平方 (省去开根号提高速度)
    dist_sq = np.sum(diff ** 2, axis=-1)  # 结果维度 (N, P, M)

    # 在 P 维度取最小值，得到每个点云离地标最近的点
    min_dist_sq = np.min(dist_sq, axis=1)  # 结果维度 (N, M)

    return np.sqrt(min_dist_sq)


def visualize_teeth_with_keypoints(mesh, labels, keypoints_dict):


    # 2. 给每颗牙齿上色 (labels: 0-16)
    # 使用高对比度离散色板，labels 为 0 的背景设为浅灰
    # 这里手动定义一个 17 色的色板
    tooth_colors = np.array([
        [128, 128, 128, 255],  # 0: 背景 (半透明浅灰)
        [255, 127, 14, 255], [31, 119, 180, 255], [44, 160, 44, 255],
        [214, 39, 40, 255], [148, 103, 189, 255], [140, 86, 75, 255],
        [227, 119, 194, 255], [0, 127, 127, 255], [188, 189, 34, 255],
        [23, 190, 207, 255], [174, 199, 232, 255], [255, 187, 120, 255],
        [152, 223, 138, 255], [255, 152, 150, 255], [197, 176, 213, 255],
        [196, 156, 148, 255]
    ], dtype=np.uint8)

    # 根据顶点 labels 赋值颜色
    mesh.visual.vertex_colors = tooth_colors[labels]

    # 3. 关键点类型颜色映射 (RGB + CMY，对人眼最敏感)
    # 这样你一眼就能分清 Mesial 和 Distal
    kp_type_colors = {
        'Mesial': [255, 0, 0, 255],  # 红色
        'Distal': [0, 255, 0, 255],  # 绿色
        'Cusp': [0, 255, 255, 255],  # 青色
        'InnerPoint': [0, 0, 255, 255],  # 蓝色
        'OuterPoint': [255, 255, 0, 255],  # 黄色
        'FacialPoint': [255, 0, 255, 255],  # 品红
    }

    visual_elements = [mesh]

    # 4. 处理 keypoints_dict
    # 遍历字典中的每一个关键点类型
    for kp_type, points in keypoints_dict.items():
        # 获取该类型对应的颜色
        color = kp_type_colors.get(kp_type, [255, 255, 255, 255])

        for pt in points:
            # 创建表示关键点的小球
            sphere = trimesh.creation.uv_sphere(radius=0.5)
            sphere.visual.face_colors = color

            # 移动到关键点坐标
            translation = np.eye(4)
            translation[:3, 3] = pt
            sphere.apply_transform(translation)

            visual_elements.append(sphere)

    # 5. 显示
    scene = trimesh.Scene(visual_elements)
    scene.bg_color = [128, 128, 128, 255]
    print("可视化就绪：牙齿按 Label 染色，关键点按类型染色。")
    scene.show()



def train_data(
    conf_thres=0.1,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    single_cls=False,  # treat as single-class dataset
):

    keypoint_path = 'G:/teethMICCAI2022/3DTeethLand_landmarks_train/'
    keyp_list =[]
    FileErgodic(keypoint_path, keyp_list, ".json")
    mesh_dict = {}
    for mesh_file in keyp_list:
        mesh_name = os.path.basename(mesh_file).replace("__kpt.json", "")
        mesh_dict[mesh_name] = mesh_file


    file_path = "H:/teethMICCAI2022/raw_data/train/"
    file_list = []
    FileErgodic(file_path, file_list, ".obj")

    # Load model
    model = torch.jit.load('../save_model/best.torchscript').cuda().eval()

    imgsz = 512
    # Configure
    model.eval()
    model3D = torch.jit.load('../save_model/tooth3D.pt')
    model3D.cuda().eval()

    save_rooth = "H:/teethMICCAI2022/seg_landmarks_train_data/train_seg_land/"
    Toothmask= np.zeros((len(file_list), 9))
    for di in range(0, len(file_list)):
        #file_list[di] = "H:/teethMICCAI2022/raw_data/train/data_part_3/lower/01A6HAN6/01A6HAN6_lower.obj"
        tic = time.time()

        gt_label_dict, pred_label_dict = {}, {}
        print(di, "   ", file_list[di])
        data_name = os.path.split(file_list[di])[-1].replace(".obj", "")

        if data_name not in mesh_dict:
            continue

        mask_path_ = file_list[di].replace(".obj", ".json")
        gtlabel, gt_instances, thids, label32 = read_teeth_mask(mask_path_)

        mesh = trimesh.load(file_list[di])
        vertics = mesh.vertices
        image, dept_img, conv_coords, row_verts, faces, row_points= get_data_to_mesh(file_list[di], [imgsz, imgsz])
        image = (image / np.max(image)).astype(np.float32)

        im = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).cuda().float()

        with torch.no_grad():

            nb, _, height, width = im.shape  # batch size, channels, height, width
            # Inference
            preds = model(im)
            preds = non_max_suppression(preds, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=single_cls,max_det=max_det)[0]
            bbox = preds[..., :4].clone()
            cls_score = preds[..., 4].clone()
            nms_index = nms(bbox, cls_score, iou_threshold=0.45)
            preds = preds[nms_index].detach().cpu().numpy()
            pcoords = np.concatenate([preds[..., :4], preds[..., -1:]+1],axis=-1)
            #pcoords = toothNumberingCorrection(pcoords)
            # pcoords --->mapping point clound
            # import cv2
            # for pi in range(pcoords.shape[0]):
            #     coord = pcoords[pi].astype(np.int32)
            #     image = cv2.drawMarker(image, ((coord[0]+ coord[2])//2, (coord[1]+ coord[3])//2), color_dict[int(coord[-1])], cv2.MARKER_DIAMOND, thickness=3)
            #     #image = cv2.rectangle(image, (coord[0], coord[1]),(coord[2], coord[3]),  color_dict[int(coord[-1])], cv2.MARKER_DIAMOND, -1)
            #     image = cv2.putText(image, str(coord[-1]), ((coord[0]+ coord[2])//2, (coord[1]+ coord[3])//2), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #                         (255, 0, 0), 2)
            # # for pi in range(gbox.shape[0]):
            # #     coord = gbox[pi].astype(np.int32)
            # #     image = cv2.circle(image, (coord[1], coord[2]), 3, (255, 0, 255), cv2.MARKER_DIAMOND, -1)
            # cv2.imwrite("../outputs/row_ing_detec_post.png", image*255)
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            # cv2.imshow("img", image)
            # cv2.waitKey(0)
        gtkey_points =None
        if data_name in mesh_dict:
            gtkey_points = read_landmarks(mesh_dict[data_name])
        #print(gtkey_points)
        #visualize_teeth_with_keypoints(mesh, label32, gtkey_points)

        teeth_points, teeth_indexes, teeth_colors, sort_tooth_dict, gtkey_points = mapping_point_clound_ellipse(pcoords, conv_coords, vertics, faces, label32, gtkey_points)

        for ti, tid in enumerate(sort_tooth_dict):
            points = sort_tooth_dict[tid][0]
            colors = sort_tooth_dict[tid][1]
            file_ = open("../outputs/tooth_"+ str(tid) + ".txt", "w")
            for point, colorv in zip(points, colors):
                file_.write(str(point[0]) + " " + str(point[1]) + " " + str(point[2]) +str(colorv[0]) + " " + str(colorv[1]) + " " + str(colorv[2]) + "\n" )
            file_.close()

        tooth_key = None
        if data_name in mesh_dict:
            sort_tooth_tid = np.array(list(sort_tooth_dict.keys())).astype(np.int32)
            all_tooth_points = {}
            for ti, tid in enumerate(sort_tooth_dict):
                points = sort_tooth_dict[tid][0]
                vl = points[:, :3][points[:, 3] >=1]
                se_index = np.random.randint(0, vl.shape[0], 1024)
                all_tooth_points[tid] = vl[se_index]

            sort_all_tooth_points = np.array([all_tooth_points[tid] for tid in sort_tooth_tid])


            tooth_key = {tid: {} for tid in sort_tooth_tid}
            tooth_key_point = {tid: [] for tid in sort_tooth_tid}
            for i, key_clss in enumerate(gtkey_points):
                gt_kp = np.array(gtkey_points[key_clss])
                #gt_kps = [{"coord":gt_kp[i],"class":key_clss}   for i in range(gt_kp.shape[0])]
                #view_row_key_point(mesh, gt_kps)

                if key_clss in ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint']:
                    # dist_cost = cdist(gt_kp, sort_tooth_cp, metric='euclidean')
                    dist_cost = compute_distance_matrix_numpy(sort_all_tooth_points, gt_kp)
                    # 2. 求解二分图最优匹配 (线性分配问题)
                    row_ind, col_ind = linear_sum_assignment(dist_cost)
                    min_dist = dist_cost[row_ind, col_ind]
                    mask = min_dist < 1
                    row_ind, col_ind = row_ind[mask], col_ind[mask]

                    for t_i in range(col_ind.shape[0]):
                        tid = sort_tooth_tid[row_ind[t_i]]
                        tooth_key[tid][key_clss] = gt_kp[col_ind[t_i]].tolist()
                        tooth_key_point[tid].append(gt_kp[col_ind[t_i]])

            gt_kp = np.array(gtkey_points["Cusp"])

            if len(gt_kp) > 0:
                dist_cost_mask = cdist(gt_kp, sort_all_tooth_points.reshape(-1, 3), metric='euclidean')
                min_dist_cost_mask = np.min(dist_cost_mask, axis=1)
                min_dist_cost_mask = min_dist_cost_mask <= 1  # 大于1mm所有cusp点不属于任何牙齿
                gt_kp = gt_kp[min_dist_cost_mask]

                minv_dist = []
                for i in range(sort_all_tooth_points.shape[0]):
                    tooth_point = sort_all_tooth_points[i]
                    if len(tooth_point) < 1:
                        tooth_point = np.array([[10000, 100000, 10000]])
                    dist_cost = cdist(gt_kp, tooth_point, metric='euclidean')

                    min_v = np.min(dist_cost, axis=1)
                    minv_dist.append(min_v)
                minv_dist = np.array(minv_dist)
                min_idx = np.argmin(minv_dist, axis=0)
                min_idx_v = np.min(minv_dist, axis=0)
                min_idx = min_idx[min_idx_v < 1]

                tid_cusp = set(sort_tooth_tid[min_idx].tolist())

                tid_cusp_kpoint = {tid: [] for tid in tid_cusp}
                for i in range(min_idx.shape[0]):
                    tid = sort_tooth_tid[min_idx[i]]
                    tid_cusp_kpoint[tid].append(gt_kp[i].tolist())

                for i, tid in enumerate(tid_cusp_kpoint):
                    cusp_kp = tid_cusp_kpoint[tid]
                    tooth_key[tid]["Cusp"] = cusp_kp



        for ti, tid in enumerate(sort_tooth_dict):
            points = sort_tooth_dict[tid][0]
            key_point = {}

            if None != tooth_key:
                nums = 0
                feat_data = tooth_key[tid]
                for i, key_lass in enumerate(feat_data):
                    kp = np.array(feat_data[key_lass]).reshape(-1, 3)
                    for ki in range(kp.shape[0]):
                        key_point[nums] = [key_lass, kp[ki]]
                        nums = nums + 1
            print(key_point)
            sort_tooth_dict[tid] = {"key_point":key_point, "points":points}
        #np.save(save_rooth + data_name + "_cv1.npy", sort_tooth_dict)

        print("over")


def test_data(
    conf_thres=0.3,  # confidence threshold

    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    single_cls=False,  # treat as single-class dataset
):

    keypoint_path = 'G:/teethMICCAI2022/3DTeethLand_landmarks_test/'
    keyp_list =[]
    FileErgodic(keypoint_path, keyp_list, ".json")
    mesh_dict = {}
    for mesh_file in keyp_list:
        mesh_name = os.path.basename(mesh_file).replace("__kpt.json", "")
        mesh_dict[mesh_name] = mesh_file


    file_path = "H:/teethMICCAI2022/raw_data/test/"
    file_list = []
    FileErgodic(file_path, file_list, ".obj")

    # Load model
    model = torch.jit.load('../save_model/best.torchscript').cuda().eval()

    imgsz = 512
    # Configure
    model.eval()
    model3D = torch.jit.load('../save_model/tooth3D.pt')
    model3D.cuda().eval()

    save_rooth = "H:/teethMICCAI2022/seg_landmarks_train_data/test/"
    Toothmask= np.zeros((len(file_list), 9))


    for di in range(0, len(file_list)):
        #file_list[di] = "H:/teethMICCAI2022/raw_data/train/data_part_3/lower/01A6HAN6/01A6HAN6_lower.obj"
        tic = time.time()

        gt_label_dict, pred_label_dict = {}, {}
        print(di, "   ", file_list[di])
        data_name = os.path.split(file_list[di])[-1].replace(".obj", "")


        mesh = trimesh.load(file_list[di])
        vertics = mesh.vertices
        image, dept_img, conv_coords, row_verts, faces, row_points= get_data_to_mesh(file_list[di], [imgsz, imgsz])
        image = (image / np.max(image)).astype(np.float32)

        im = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).cuda().float()

        with torch.no_grad():

            nb, _, height, width = im.shape  # batch size, channels, height, width
            # Inference
            preds = model(im)
            preds = non_max_suppression(preds, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=single_cls,max_det=max_det)[0]
            bbox = preds[..., :4].clone()
            cls_score = preds[..., 4].clone()
            nms_index = nms(bbox, cls_score, iou_threshold=0.45)
            preds = preds[nms_index].detach().cpu().numpy()
            pcoords = np.concatenate([preds[..., :4], preds[..., -1:]+1],axis=-1)

            ##Tooth Numbering Correction
            pcoords = toothNumberingCorrection(pcoords)
            # pcoords --->mapping point clound
            # import cv2
            # for pi in range(pcoords.shape[0]):
            #     coord = pcoords[pi].astype(np.int32)
            #     image = cv2.drawMarker(image, ((coord[0]+ coord[2])//2, (coord[1]+ coord[3])//2), (255, 0, 255), cv2.MARKER_DIAMOND, thickness=3)
            #     image = cv2.rectangle(image, (coord[0], coord[1]),(coord[2], coord[3]),  color_dict[int(coord[-1])], cv2.MARKER_DIAMOND, -1)
            #     image = cv2.putText(image, str(coord[-1]), ((coord[0]+ coord[2])//2, (coord[1]+ coord[3])//2), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #                         (255, 0, 0), 2)
            # # for pi in range(gbox.shape[0]):
            # #     coord = gbox[pi].astype(np.int32)
            # #     image = cv2.circle(image, (coord[1], coord[2]), 3, (255, 0, 255), cv2.MARKER_DIAMOND, -1)
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            # cv2.imshow("img", image)
            # cv2.waitKey(0)



        teeth_points, sort_tooth_dict = test_mapping_point_clound_ellipse(pcoords, conv_coords, vertics, faces)
        # save_path = "../outputs/" + str(data_name) + "/"
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        # for ti, tid in enumerate(sort_tooth_dict):
        #     points = sort_tooth_dict[tid][0]
        #     file_ = open(save_path + "/tooth_"+ str(tid) + ".txt", "w")
        #     colorv = color_dict[tid]
        #     for point in points:
        #         file_.write(str(point[0]) + " " + str(point[1]) + " " + str(point[2]) +str(colorv[0]) + " " + str(colorv[1]) + " " + str(colorv[2]) + "\n" )
        #     file_.close()

        tooth_key = None
        if data_name in mesh_dict:
            feat_data = read_landmarks(mesh_dict[data_name])

            nums = 0
            tooth_key = {}
            for i, key_lass in enumerate(feat_data):
                kp = np.array(feat_data[key_lass]).reshape(-1, 3)
                for ki in range(kp.shape[0]):
                    tooth_key[nums] = [key_lass, kp[ki]]
                    nums = nums + 1


        for ti, tid in enumerate(sort_tooth_dict):
            points = sort_tooth_dict[tid][0]
            key_point = {}

            if None != tooth_key:
                key_point = tooth_key

            sort_tooth_dict[tid] = {"key_point":key_point, "points":points}
        np.save(save_rooth + data_name + "_cv.npy", sort_tooth_dict)

        print("over")


if __name__ == "__main__":
    train_data()

    #test_data()
