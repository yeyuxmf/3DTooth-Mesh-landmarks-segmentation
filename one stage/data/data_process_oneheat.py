import os
import sys
import time
import vtk
import json
import cv2
import trimesh
from torchvision.ops import nms
import numpy as np
import pyvista as pv
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch


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

color_space = {
    0: (119, 119, 119),  # 背景/未知 (保留灰色)

    # --- 上颌 (1-16) 对称组 ---
    1: (230, 25, 75), 16: (230, 25, 75),  # 鲜红
    2: (60, 180, 75), 15: (60, 180, 75),  # 翠绿
    3: (255, 225, 25), 14: (255, 225, 25),  # 亮黄
    4: (0, 130, 200), 13: (0, 130, 200),  # 蔚蓝
    5: (245, 130, 48), 12: (245, 130, 48),  # 橙色
    6: (145, 30, 180), 11: (145, 30, 180),  # 紫色
    7: (70, 240, 240), 10: (70, 240, 240),  # 青色
    8: (240, 50, 230), 9: (240, 50, 230),  # 洋红

    # --- 下颌 (17-32) 对称组 ---
    17: (210, 245, 60), 32: (210, 245, 60),  # 莱姆绿
    18: (250, 190, 212), 31: (250, 190, 212),  # 浅粉
    19: (0, 128, 128), 30: (0, 128, 128),  # 深青
    20: (220, 190, 255), 29: (220, 190, 255),  # 薰衣草紫
    21: (170, 110, 40), 28: (170, 110, 40),  # 褐色
    22: (128, 0, 0), 27: (128, 0, 0),  # 栗色
    23: (170, 255, 195), 26: (170, 255, 195),  # 薄荷绿
    24: (128, 128, 0), 25: (128, 128, 0)  # 橄榄绿
}
CONVERT_TABLE = {
    0: 0, 18: 1, 17: 2, 16: 3, 15: 4, 14: 5, 13: 6, 12: 7, 11: 8,
    21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
    38: 17, 37: 18, 36: 19, 35: 20, 34: 21, 33: 22, 32: 23, 31: 24,
    41: 25, 42: 26, 43: 27, 44: 28, 45: 29, 46: 30, 47: 31, 48: 32
}


CLAS_INDEX = {0 :"0",  8: '1', 7: '2', 6: '3', 5: '4', 4: '5', 3: '6', 2: '7', 1: '8',
                 9: '1', 10: '2', 11: '3', 12: '4', 13: '5', 14: '6', 15: '7', 16: '8',
                 24: '1', 23: '2', 22: '3', 21: '4', 20: '5', 19: '6', 18: '7', 17: '8',
                 25: '1', 26: '2', 27: '3', 28: '4', 29: '5', 30: '6', 31: '7', 32: '8'}

CLAS_RLINDEX = {8: '0', 7: '0', 6: '0', 5: '0', 4: '0', 3: '0', 2: '0', 1: '0',
                 9: '1', 10: '1', 11: '1', 12: '1', 13: '1', 14: '1', 15: '1', 16: '1',
                 24: '0', 23: '0', 22: '0', 21: '0', 20: '0', 19: '0', 18: '0', 17: '0',
                 25: '1', 26: '1', 27: '1', 28: '1', 29: '1', 30: '1', 31: '1', 32: '1'}

tooth_colors = np.array([
    [128, 128, 128, 255],  # 0: 背景 (半透明浅灰)
    [255, 127, 14, 255], [31, 119, 180, 255], [44, 160, 44, 255],
    [214, 39, 40, 255], [148, 103, 189, 255], [140, 86, 75, 255],
    [227, 119, 194, 255], [0, 127, 127, 255], [188, 189, 34, 255],
    [23, 190, 207, 255], [174, 199, 232, 255], [255, 187, 120, 255],
    [152, 223, 138, 255], [255, 152, 150, 255], [197, 176, 213, 255],
    [196, 156, 148, 255],
    [255, 127, 14, 255], [31, 119, 180, 255], [44, 160, 44, 255],
    [214, 39, 40, 255], [148, 103, 189, 255], [140, 86, 75, 255],
    [227, 119, 194, 255], [0, 127, 127, 255], [188, 189, 34, 255],
    [23, 190, 207, 255], [174, 199, 232, 255], [255, 187, 120, 255],
    [152, 223, 138, 255], [255, 152, 150, 255], [197, 176, 213, 255],
    [196, 156, 148, 255]
], dtype=np.uint8)


def FileErgodic(file_root, flie_list, type_):

    for file in os.listdir(file_root):
      newDir = os.path.join(file_root,file)
      if os.path.isdir(newDir):
          FileErgodic(newDir, flie_list, type_)
      else:
        if type_ in file:
            flie_list.append(newDir)

def read_teeth_mask(file_name):
    with open(file_name, "r") as file_:
        mask_json = json.load(file_)
        label = np.array(mask_json["labels"])
        instance = np.array(mask_json["instances"])

    nonv = np.unique(label[np.nonzero(label)])
    thids = {}
    for i in range(nonv.shape[0]):

        tindex = np.where(label == nonv[i])
        tid = CONVERT_TABLE[label[tindex[0]][0]]
        thids[(i, int(tid))] = tindex[0]

    label32 = np.array([CONVERT_TABLE[tid]   for tid in label])

    return label, instance, thids, label32


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
        [196, 156, 148, 255],
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



import matplotlib.colors as mcolors
def visualize_mesh_with_landmarks(tri_mesh, tooth_info):
    # 1. 初始化 PyVista 渲染器，并设置分辨率 (等同于 viewer_kwargs)
    plotter = pv.Plotter(window_size=[784, 784])

    # 2. 将 trimesh 对象直接转换为 pyvista 对象，并添加到渲染器中
    # 默认颜色为白色，你可以通过 color='white' 或传入 RGB 指定基础网格颜色
    pv_mesh = pv.wrap(tri_mesh)
    plotter.add_mesh(pv_mesh, color='#D3D3D3', opacity=1.0, smooth_shading=True, label="Teeth Mesh")
    # 3. 遍历并添加关键点球体
    for i, tid in enumerate(tooth_info):
        tooth_kp, _ = tooth_info[tid]

        for kp_name, kp_pos in tooth_kp.items():
            kp_pos = np.array(kp_pos).reshape(-1, 3)
            # short_name = kp_name.split("_")[0] # 原代码中未使用此变量，若不需要可删除

            # 获取颜色：注意 PyVista 的 RGB 列表通常接受 0.0 到 1.0 的浮点数
            # 如果你的 color_space 是 0-255 范围，需要除以 255
            point_color = color_space.get(tid, [255, 255, 255])
            pv_color = [c / 255.0 for c in point_color]

            for ki in range(kp_pos.shape[0]):
                # 直接在指定中心点生成球体，省去了矩阵平移的步骤
                sphere = pv.Sphere(radius=0.8, center=kp_pos[ki])

                # 将球体加入渲染器并赋色
                plotter.add_mesh(sphere, color=pv_color)

    # 4. 显示场景
    plotter.view_xy()
    plotter.show()

def data_view(mesh, tooth_info):

    visual_elements = [mesh]
    for i, tid in enumerate(tooth_info):
        _, tooth_kp, _ = tooth_info[tid]

        for kp_name, kp_pos in tooth_kp.items():
            kp_pos = np.array(kp_pos).reshape(-1, 3)
            short_name = kp_name.split("_")[0]

            point_color = color_space.get(tid, [255, 255, 255])



            for ki in range(kp_pos.shape[0]):
                sphere = trimesh.creation.uv_sphere(radius=0.5)
                sphere.visual.face_colors = point_color
                translation = np.eye(4)
                translation[:3, 3] = kp_pos[ki]
                sphere.apply_transform(translation)
                visual_elements.append(sphere)

    scene = trimesh.Scene(visual_elements)
    scene.show(viewer_kwargs={'resolution': (512, 512)})


def farthest_point_sample_np(
        points: np.ndarray,
        K: int,
        random_start: bool = True
) -> np.ndarray:
    # 1. 严格校验输入形状和参数
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"点云形状必须为 (N, 3)，当前输入形状为 {points.shape}")
    N = points.shape[0]
    if K > N:
        raise ValueError(f"采样数K={K} 不能大于原始点数N={N}")
    if K <= 0:
        raise ValueError(f"采样数K={K} 必须为正整数")

    # 2. 初始化：采样索引数组 + 最小距离数组
    sampled_indices = np.zeros(K, dtype=np.int32)
    # min_distances[i] = 点i到已选点集的最小距离（初始为无穷大）
    min_distances = np.full(N, np.inf)

    # 3. 选择第一个点
    if random_start:
        start_idx = np.random.randint(0, N)  # 随机选起始点
    else:
        start_idx = 0  # 固定选第一个点
    sampled_indices[0] = start_idx

    # 4. 核心迭代：逐次选择最远点（向量化加速）
    for k in range(1, K):
        # 获取上一轮选中的点 (3,)
        last_selected = points[sampled_indices[k - 1]]
        # 向量化计算所有点到上一轮选点的欧氏距离平方 (N,)
        dist = np.sum((points - last_selected) ** 2, axis=1)
        # 更新最小距离：取「当前距离」和「历史最小距离」的较小值
        min_distances = np.minimum(min_distances, dist)
        # 选择距离最大的点作为下一个采样点
        sampled_indices[k] = np.argmax(min_distances)

    return sampled_indices

def  FurthestPointSampling(xyz, npoint):
    import pointnet2_ops._ext as _ext
    xyz = torch.tensor(xyz).unsqueeze(dim=0).cuda().float()
    index = _ext.furthest_point_sampling(xyz, npoint).squeeze().detach().cpu().numpy()

    return index
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


def get_new_tooth_index(vertics, index, tooth_info):
    """
    index: FPS 采样后保留在原 vertics 中的索引列表 (形状为 [N_sampled])
    tooth_info: 字典 {tid: [points, key_point, old_idx]}
    """
    new_tooth_info = {}

    # 1. 初始化全为 -1 的 mask (np.full 比 np.ones * -1 更快更直接)
    mask = np.full((vertics.shape[0],), -1, dtype=np.int32)

    # 2. 一步到位建立映射: 将采样保留的 index 映射到新的顺序 0, 1, 2...
    mask[index] = np.arange(len(index), dtype=np.int32)
    colors = np.zeros((index.shape[0]), np.float32)
    for tid, info in tooth_info.items():
        key_point, idx = info

        # 3. 批量映射并过滤
        new_idx = mask[idx]
        new_idx = new_idx[new_idx >= 0]

        # 4. 可选的安全校验：如果某颗牙的点被全部下采样掉了，就不存入新字典
        if len(new_idx) > 0:
            new_tooth_info[tid] = [key_point, new_idx]
        colors[new_idx] = tid
    return new_tooth_info, colors







def train_data():

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


    save_rooth = 'G:/teethMICCAI2022/train_land_onenet_data/trainoneheat/'

    for di in range(0, len(file_list)):

        print(di, "   ", file_list[di])
        data_name = os.path.split(file_list[di])[-1].replace(".obj", "")

        if data_name not in mesh_dict:
            continue

        cenp_landmarks_path = file_list[di].replace(".obj", "_private.npy")
        mask_path_ = file_list[di].replace(".obj", ".json")
        gtlabel, gt_instances, thids, label32 = read_teeth_mask(mask_path_)

        sys.modules['numpy._core'] = sys.modules['numpy.core']
        sys.modules['numpy._core.multiarray'] = sys.modules['numpy.core.multiarray']
        cenp_landmarks = np.load(cenp_landmarks_path, allow_pickle=True).item()

        mesh = trimesh.load(file_list[di])
        vertics = mesh.vertices


        tooth_info = {}
        for i, id in enumerate(thids):
            tid = id[1]
            index = thids[id]
            points = vertics[index]
            tooth_info[tid] = [points, cenp_landmarks[tid][1], index]

        tooth_info = {k: tooth_info[k] for k in sorted(tooth_info)}

        keys = np.array(list(tooth_info.keys()))
        tooth_points = []
        for i, key in enumerate(tooth_info):
            tpoints = tooth_info[key][0]
            if tpoints.shape[0] > 1024:
                index = FurthestPointSampling(tpoints.astype(np.float32), npoint=1024)
            else:
                index = np.random.randint(0, tpoints.shape[0], 1024)
            tooth_points.append(tpoints[index])
        tooth_points = np.array(tooth_points)


        gtkey_points = {}
        if data_name in mesh_dict:
            gtkey_points = read_landmarks(mesh_dict[data_name])

            #print(gtkey_points)
            #visualize_teeth_with_keypoints(mesh, label32, gtkey_points)

            tooth_data = defaultdict(lambda: defaultdict(list))
            THR = 1.0  # 距离阈值

            for clss, gt_kp in gtkey_points.items():
                if len(gt_kp) == 0: continue
                gt_kp = np.array(gt_kp)
                dist_mat = compute_distance_matrix_numpy(tooth_points, gt_kp)

                # 1. 确定匹配索引 (p_idx 对应 tooth_points, g_idx 对应 gt_kp)
                if clss == 'Cusp':
                    p_idx = np.argmin(dist_mat, axis=0)  # 每个GT找最近的预测牙齿
                    g_idx = np.arange(len(gt_kp))
                else:
                    p_idx, g_idx = linear_sum_assignment(dist_mat)  # 一对一分配

                # 2. 过滤并存储
                for pi, gi in zip(p_idx, g_idx):
                    if dist_mat[pi, gi] < THR:
                        tid = keys[pi]
                        tooth_data[tid][clss].append(gt_kp[gi])

            # 如果你想转回普通字典
            tooth_data = dict(tooth_data)
            ou_tooth_data = {}
            #['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'Cusp']
            for ki, tid in enumerate(tooth_data):
                kp = tooth_data[tid]
                # if 'Mesial' not in kp:
                #     kp['Mesial'] = tooth_info[tid][1]['MCP']
                # if 'Distal' not in kp:
                #     kp['Distal'] = tooth_info[tid][1]['DCP']
                # if 'InnerPoint' not in kp:
                #     kp['InnerPoint'] = tooth_info[tid][1]['InnerPoint']
                # if 'OuterPoint' not in kp:
                #     kp['OuterPoint'] = tooth_info[tid][1]['OuterPoint']
                # if 'FacialPoint' not in kp:
                #     kp['FacialPoint'] = tooth_info[tid][1]['FA']

                if "DeCP" in  tooth_info[tid][1]:
                    cenp = tooth_info[tid][1]['DeCP']
                if "CEP" in tooth_info[tid][1]:
                    cenp = tooth_info[tid][1]['CEP']
                if "CFP" in tooth_info[tid][1]:
                    cenp = tooth_info[tid][1]['CFP']
                kp['cenpland'] = cenp.reshape(1, 3)

                ou_tooth_data[tid] = [kp, tooth_info[tid][2].astype(np.int32)]

            #部分牙齿只有分割
            remove_tid = list(set(list(tooth_info.keys())) - set(list(ou_tooth_data.keys())))
            for reitid in remove_tid:
                kep = {}
                if "DeCP" in  tooth_info[reitid][1]:
                    cenp = tooth_info[reitid][1]['DeCP']
                if "CEP" in tooth_info[reitid][1]:
                    cenp = tooth_info[reitid][1]['CEP']
                if "CFP" in tooth_info[reitid][1]:
                    cenp = tooth_info[reitid][1]['CFP']
                kep["cenpseg"] = np.array(list(cenp))
                kep['InnerPoint'] = tooth_info[reitid][1]['InnerPoint']
                kep['OuterPoint'] = tooth_info[reitid][1]['OuterPoint']
                kep['Mesial'] = tooth_info[reitid][1]['MCP']
                kep['Distal'] = tooth_info[reitid][1]['DCP']
                ou_tooth_data[reitid] = [kep, tooth_info[reitid][2].astype(np.int32)]


        if vertics.shape[0]>8192*2:
            index = FurthestPointSampling(vertics.astype(np.float32), npoint=8192 * 2)
            #index = farthest_point_sample_np(vertics.astype(np.float32), K=8192*2)
        else:
            index = np.random.randint(0, vertics.shape[0], 8192*2)

        ou_tooth_data, colors = get_new_tooth_index(vertics, index, ou_tooth_data)




        tooth_data = {"Data":vertics[index], "landmarks":ou_tooth_data}
        np.save(save_rooth + data_name + ".npy", tooth_data)
        #visualize_mesh_with_landmarks(mesh, ou_tooth_data)
        #data_view(mesh, tooth_info)

        # file_ = open("../outputs/data_points.txt", "w")
        # data_points = vertics[index]
        # for i in range(data_points.shape[0]):
        #     point = data_points[i]
        #     clolov = tooth_colors[int(colors[i])]
        #     file_.write(str(point[0])+" " + str(point[1])+" " +str(point[2])+" " +str(clolov[0])+" " + str(clolov[1])+" " +str(clolov[2])+" \n")
        # file_.close()


        #np.save(save_rooth + data_name + "_c.npy", tooth_data)

        print("over")


def test_data():

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


    save_rooth = 'G:/teethMICCAI2022/train_land_onenet_data/test/'
    Toothmask= np.zeros((len(file_list), 9))
    for di in range(0, len(file_list)):


        print(di, "   ", file_list[di])
        data_name = os.path.split(file_list[di])[-1].replace(".obj", "")

        if data_name not in mesh_dict:
            continue

        mask_path_ = file_list[di].replace(".obj", ".json")

        label32 = np.array([1])
        if os.path.exists(mask_path_):
            gtlabel, gt_instances, thids, label32 = read_teeth_mask(mask_path_)

        mesh = trimesh.load(file_list[di])
        vertics = mesh.vertices


        gtkey_points =None
        if data_name in mesh_dict:
            gtkey_points = read_landmarks(mesh_dict[data_name])
        #print(gtkey_points)
            visualize_teeth_with_keypoints(mesh, label32, gtkey_points)

        if vertics.shape[0]>8192*1:
            index = FurthestPointSampling(vertics.astype(np.float32), npoint=8192*1)
            #index = farthest_point_sample_np(vertics.astype(np.float32), K=8192*2)
        else:
            index = np.random.randint(0, vertics.shape[0], 8192*1)

        sort_tooth_dict ={}
        sort_tooth_dict["index"] = index.astype(np.int32)
        sort_tooth_dict["points"] = vertics.astype(np.float32)
        sort_tooth_dict["landmarks"] = gtkey_points
        sort_tooth_dict["segment"] = label32.astype(np.int32)

        #np.save(save_rooth + data_name + "_c.npy", sort_tooth_dict)

        print("over")



if __name__ == "__main__":
    train_data()

    #test_data()
