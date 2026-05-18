
import os
import copy
import torch
import numpy as np
from data.util import rotate_maxtrix
from config import config as cfg
from scipy.spatial import KDTree
from scipy.spatial import cKDTree

def get_files(file_dir, file_list, type_str):

    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(path, file_list, type_str)
        else:
            if file_.rfind(type_str) !=-1:
                file_list.append(path)

def st_random_axis_rotate(data_points, landmarks):

    v1 = np.sign(np.random.normal(0, 1, size=(1))[0])
    rotaxis = np.random.random(3) * 2 - 1 + 0.01
    rotaxis = rotaxis / np.linalg.norm(rotaxis)

    angle_ = v1 * cfg.dAngles[np.random.randint(0, len(cfg.dAngles), 1)[0]]  # [-10°--10°]
    rt = rotate_maxtrix(rotaxis, angle_)
    rt = rt[0:3, 0:3]
    cp = np.mean(data_points, axis=0, keepdims=True)
    data_points = data_points - cp

    data_points = (rt.dot(data_points.T)).T + cp

    if len(landmarks) >=1:
        landmarks = landmarks - cp
        landmarks = (rt.dot(landmarks.T)).T + cp


    return data_points, landmarks

def st_random_axis_trans(data_points, landmarks, scalev=1.0):
    trans_v = np.array([[-1, -1, 1]]) * scalev

    v1 = np.random.normal(0, 1, size=(1))[0]
    v2 = np.random.normal(0, 1, size=(1))[0]
    v3 = np.random.normal(0, 1, size=(1))[0]
    fg = np.clip(np.array([[v1, v2, v3]]), -1, 1)
    data_points = data_points + fg * trans_v

    if len(landmarks) >= 1:
        landmarks = landmarks + fg * trans_v

    return data_points, landmarks
def random_remove_tooth(file_data):
    key = np.array(list(file_data.keys()))
    nums = key.shape[0]
    if nums<= cfg.max_remove_nums+4:

        return file_data

    rotate_nums = np.random.randint(0, cfg.max_remove_nums, 1)[0]
    rotate_index = [i for i in range(nums)]
    np.random.shuffle(rotate_index)
    select_index = rotate_index[rotate_nums:]
    select_key = key[select_index]

    file_data_ = { tid:  file_data[tid] for i, tid in enumerate(file_data) if tid in select_key}

    return file_data_

# def read_data(file_path):
#
#
#     file_data = np.load(file_path, allow_pickle=True).item()
#
#     teeth_landmarks = []
#     tooth_mask = np.zeros((cfg.tooth_nums, max(cfg.tooth_landmark_nums)), np.int32)
#     #tooth_mask = np.zeros((cfg.tooth_nums), np.int32)
#     data_points = np.zeros((cfg.tooth_nums, cfg.sam_points, 3), np.int32)
#
#     # random remove tooth
#     file_data = random_remove_tooth(file_data)
#
#     teeth_nums = []
#     for key in file_data:
#
#         landmarks_ = file_data[key][0]
#
#         mesh_points = np.array(file_data[key][1])
#         se_index = np.random.randint(0, mesh_points.shape[0], cfg.sam_points)
#         select_points = mesh_points[se_index]
#
#         if len(landmarks_)>cfg.maxlandnums:
#             continue
#
#         lcls, lcoord = [], []
#         for i in range(len(landmarks_)):
#             cls, coord = landmarks_[i]
#             lcls.append(cls)
#             lcoord.append(coord)
#
#         sorted_dict = np.array([lcoord[lcls.index(k)] for k in cfg.fxiedPorder if k in lcls])
#
#
#         if 0 == np.random.randint(0, 3, 1):
#             select_points, sorted_dict = st_random_axis_rotate(select_points, np.array(sorted_dict))
#
#         data_points[int(key) - 1] = select_points
#
#         sorted_key_idx = [cfg.fxiedPorder.index(k) for k in cfg.fxiedPorder if k in lcls]
#         tooth_mask[int(key) - 1, sorted_key_idx] = 1
#
#         #tooth_mask[int(key) - 1] = len(sorted_dict)
#
#         teeth_landmarks.append(sorted_dict)
#         teeth_nums.append(key)
#
#     teeth_nums = np.array(teeth_nums)
#     order_index = np.argsort(teeth_nums)
#     teeth_landmarks = np.concatenate([teeth_landmarks[ki] for ki in order_index], axis=0)
#
#     return data_points, teeth_landmarks, tooth_mask

def read_data(file_path):


    file_data = np.load(file_path, allow_pickle=True).item()

    teeth_landmarks = np.zeros((cfg.tooth_nums, cfg.fxid_tnums + cfg.match_tnums, 3), np.float32)
    tooth_mask = np.zeros((cfg.tooth_nums), np.int32)
    land_mask = np.zeros((cfg.tooth_nums, cfg.fxid_tnums + cfg.match_tnums), np.int32)
    data_points = np.zeros((cfg.tooth_nums, cfg.sam_points, 3), np.float64)
    data_label = np.zeros((cfg.tooth_nums, cfg.sam_points, 1), np.float64)
    # random remove tooth
    file_data = random_remove_tooth(file_data)

    land_flag = False
    for key in file_data:

        landmarks_ = file_data[key]["key_point"]
        mesh_points = np.array(file_data[key]["points"])

        if mesh_points.shape[0] > cfg.sam_points:
            se_index = np.random.choice(mesh_points.shape[0], size=cfg.sam_points, replace=False)
        else:
            se_index = np.random.randint(0, mesh_points.shape[0], cfg.sam_points)
        # se_index = file_data[key]["farthestindex"][2]
        select_points = mesh_points[se_index][:, :3]
        select_labels = mesh_points[se_index][:, 3:]

        if len(landmarks_)>cfg.maxlandnums:
            continue

        merge_dict = {}
        if len(landmarks_)>=1:
            lcls, lcoord = [], []
            for i in range(len(landmarks_)):
                cls, coord = landmarks_[i]
                lcls.append(cls)
                lcoord.append(coord)

            sorted_dict = np.array([lcoord[lcls.index(k)] for k in cfg.fxiedPorder if k in lcls]).reshape(-1, 3)
            match_dict = np.array([lcoord[mi] for mi in range(len(lcls)) if lcls[mi] in cfg.matchPorder]).reshape(-1, 3)
            merge_dict = np.concatenate([sorted_dict, match_dict], axis=0)


        if 0 == np.random.randint(0, 3, 1):
            select_points, merge_dict = st_random_axis_rotate(select_points, merge_dict)

        if 0 == np.random.randint(0, 3, 1):
            select_points, merge_dict = st_random_axis_trans(select_points, merge_dict)
        if 0 == np.random.randint(0, 3, 1):
            p_normals = np.array([[1, 1, 1]])
            select_points, merge_dict, _, _ = augment_non_uniform_scaling(select_points, merge_dict, p_normals=p_normals, p_curv=p_normals, scale_range=(0.8, 1.2))


        if len(landmarks_) >= 1:
            sorted_dict = merge_dict[:sorted_dict.shape[0], :]
            match_dict = merge_dict[sorted_dict.shape[0]:, :]
            sorted_key_idx = [cfg.fxiedPorder.index(k) for k in cfg.fxiedPorder if k in lcls]
            match_key_idx = [i+cfg.fxid_tnums for i in range(match_dict.shape[0])]
            land_mask[int(key) - 1, sorted_key_idx] = 1
            land_mask[int(key) - 1, match_key_idx] = 1
            teeth_landmarks[int(key) - 1, sorted_key_idx, :] = sorted_dict
            teeth_landmarks[int(key) - 1, match_key_idx, :] = match_dict
            land_flag = True

        tooth_mask[int(key) - 1] =1
        data_points[int(key) - 1] = select_points[:, :3]
        data_label[int(key) - 1] = select_labels

    return data_points, data_label, teeth_landmarks, land_mask, tooth_mask, land_flag


def random_axis_rotate(data_points, landmarks):

    v1 = np.sign(np.random.normal(0, 1, size=(1))[0])
    rotaxis = np.random.random(3) * 2 - 1 + 0.01
    rotaxis = rotaxis / np.linalg.norm(rotaxis)

    angle_ = v1 * cfg.Angles[np.random.randint(0, len(cfg.Angles), 1)[0]]  # [-10°--10°]
    rt = rotate_maxtrix(rotaxis, angle_)
    rt = rt[0:3, 0:3]

    TN, PN, CN = data_points.shape
    data_points = data_points.reshape(TN*PN, CN)
    data_points = (rt.dot(data_points.T)).T

    TNl, PNl, CNl = landmarks.shape
    landmarks = landmarks.reshape(TNl*PNl, CNl)
    landmarks = (rt.dot(landmarks.T)).T

    data_points = data_points.reshape(TN, PN, CN)
    landmarks = landmarks.reshape(TNl, PNl, CNl)

    return data_points, landmarks

def whole_non_uniform_scaling(data_points, landmarks, scale_range=(0.8, 1.2)):




    scales = np.random.uniform(scale_range[0], scale_range[1], size=3).reshape(1, 1, 3)
    # 2. 坐标点缩放: P' = P * S
    TN, PN, CN = data_points.shape

    p_points_aug = data_points * scales
    landmarks_aug = landmarks * scales

    return p_points_aug, landmarks_aug

def assign_colors_by_probability(points, probabilities):
    """根据概率值赋予蓝色渐变（浅蓝→深蓝）"""
    probabilities = np.clip(probabilities, 0, 1).flatten()
    colors = np.zeros((len(probabilities), 3))
    colors[:, 2] = np.round(255 * probabilities)  # 蓝色通道随概率增加
    colors[:, 0] = np.round(255 * (1 - probabilities))  # 红色通道递减（可选）
    colors[:, 1] = np.round(255 * (1 - probabilities))  # 绿色通道递减（可选）
    return np.hstack((points, colors.astype(int)))
def save_colored_points(filename, colored_points):
    # 确保RGB是整数
    colored_points[:, 3:] = np.round(colored_points[:, 3:])

    # 保存为txt文件，使用空格分隔
    np.savetxt(filename, colored_points, fmt='%.6f %.6f %.6f %d %d %d')

class TrainData():
    def __init__(self, file_root):
        self.data_dir = file_root

        self.train_list = None
        self.prepare(self.data_dir)

    def prepare(self, file_path):

        file_list = []
        get_files(file_path, file_list, "npy")

        self.train_list = file_list

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, item):

        file_path = self.train_list[item]

        data_points,  data_label, teeth_landmarks, land_mask, tooth_mask, land_flag = read_data(file_path)


        #Decentering data
        cpoint = np.mean(data_points[tooth_mask >=1].reshape(-1, 3), axis=0, keepdims=True)
        data_points[tooth_mask >=1] = data_points[tooth_mask >=1] - cpoint.reshape(1, 1, 3)
        data_points[tooth_mask < 1] = np.random.normal(loc=0.0, scale=1.0, size=(cfg.sam_points, 3))
        teeth_landmarks[tooth_mask >=1]  = teeth_landmarks[tooth_mask >=1]  - cpoint


        #random axis rotate
        data_points, teeth_landmarks = random_axis_rotate(data_points, teeth_landmarks)
        data_points, teeth_landmarks = whole_non_uniform_scaling(data_points, teeth_landmarks, scale_range=(0.8, 1.2))



        heat_map = np.zeros((cfg.tooth_nums, cfg.sam_points, 2), np.float32)
        if land_flag:
            heat_map = get_hotmap(data_points, teeth_landmarks, land_mask)
            # for i in range(heat_map.shape[-1]):
            #     data_points_ = data_points.reshape(-1, 3)
            #     probabilities = heat_map[..., i].reshape(-1)
            #     colored_points = assign_colors_by_probability(data_points_, probabilities)
            #     save_colored_points("./outputs/heat_points" + str(i) + ".txt", colored_points)

        #pcurvature = get_calculate_curvature(data_points, tooth_mask)


        data_points = torch.tensor(data_points)
        land_mask = torch.tensor(land_mask)
        data_label = torch.tensor(data_label)
        tooth_mask = torch.tensor(tooth_mask)
        teeth_landmarks = torch.tensor(teeth_landmarks)
        heat_map = torch.tensor(heat_map)

        return data_points, data_label, tooth_mask, land_mask, teeth_landmarks, heat_map

def get_hotmap(data_points, teeth_landmarks, tooth_point_mask, sigma = 0.7):
    mask = np.sum(tooth_point_mask, axis=-1)
    index = np.where(mask >= 1)[0]

    heat_map = np.zeros((cfg.tooth_nums, cfg.sam_points, 2), np.float32)
    for id in index:
        points = data_points[id]
        landmarks = teeth_landmarks[id]
        mask = tooth_point_mask[id] > 0

        mask5 = mask[:5]
        mask_cusp = mask[5:]
        indices, distances = find_nearest_k_points(points, landmarks[mask], k=cfg.neark)

        knums = cfg.neark*np.sum(mask5)
        kindices  = indices[:knums]
        kgaussian_weights = np.exp(-(distances[:knums] ** 2) / (2 * sigma ** 2))
        heat_map[id, kindices, 0] = kgaussian_weights

        ############cusp#################
        indices = indices[knums:]
        distances = distances[knums:]

        if len(indices)>=1*cfg.neark:

            sort_idx = np.argsort(distances)
            indices_sorted = indices[sort_idx]
            distances_sorted = distances[sort_idx]

            # 2. 找到每个索引第一次出现的位置（即最小距离所在位置）
            _, unique_first_indices = np.unique(indices_sorted, return_index=True)

            # 3. 提取不重复索引及其对应的最小距离
            final_indices = indices_sorted[unique_first_indices]
            final_distances = distances_sorted[unique_first_indices]

            gaussian_weights = np.exp(-(final_distances ** 2) / (2 * sigma ** 2))
            # gaussian_weights = np.sort(gaussian_weights)
            # 4. 填充 heat_map
            heat_map[id, final_indices, 1] = gaussian_weights

    return heat_map

def find_nearest_k_points(points, landmarks, k=5):
    """
    在 points 中寻找离 landmarks 中每个点最近的 K 个点
    """
    # 1. 构建 KDTree (对海量点云进行空间索引)
    tree = KDTree(points)

    # 2. 查询最近邻
    # distances: 每个 landmark 到其 K 个近邻的欧式距离
    # indices: 这 K 个近邻在原始 points 数组中的索引
    distances, indices = tree.query(landmarks, k=k)

    distances = distances.ravel()
    indices = indices.ravel()
    heat_map = np.zeros((points.shape[0], 1))
    # 3. 提取具体的点坐标
    # neighbors_pts 的形状为 (len(landmarks), K, 3)


    return indices, distances


def get_calculate_curvature(data_points, tooth_point_mask):

    mask = tooth_point_mask
    index = np.where(mask>=1)[0]

    pcurvature = np.zeros((cfg.tooth_nums, cfg.sam_points, 3), np.float32)
    for id in index:
        points = data_points[id]
        curvature = calculate_signed_curvature(points, k=15)
        pcurvature[id, :, :] = (curvature.reshape(-1, 1) *20).clip(-10, 10)


    return pcurvature

def calculate_curvature(points, k=15):
    """
    快速计算点云曲率
    points: (N, 3) 的 numpy 数组
    k: 邻域点数量 (正畸模型建议 15-30)
    """

    tree = cKDTree(points)
    # k+1 是因为搜索时会包含点本身
    _, idxs = tree.query(points, k=k + 1)

    # 获取邻域点坐标 (N, k+1, 3)
    neighbors = points[idxs]

    # 1. 中心化邻域点 (减去邻域质心)
    centroids = np.mean(neighbors, axis=1, keepdims=True)  # (N, 1, 3)
    centered_neighbors = neighbors - centroids  # (N, k+1, 3)

    # 2. 计算每个点的协方差矩阵 (N, 3, 3)
    # 矩阵公式: C = (1/k) * Σ(P - P_mean).T * (P - P_mean)
    # 使用 einsum 实现批量矩阵乘法
    cov = np.einsum('nij,nik->njk', centered_neighbors, centered_neighbors) / k

    # 3. 批量计算特征值
    # linalg.eigvalsh 专门用于对称矩阵（Hessian/Covariance），速度比 eig 快得多
    eigenvalues = np.linalg.eigvalsh(cov)  # 返回值已按升序排列 (N, 3)

    # 4. 计算曲率 sigma = λ0 / (λ0 + λ1 + λ2)
    lambda_sum = np.sum(eigenvalues, axis=1)
    # 避免除以零
    curvature = eigenvalues[:, 0] / (lambda_sum + 1e-8)

    return curvature


def calculate_signed_curvature(points, k=15):
    tree = cKDTree(points)
    _, idxs = tree.query(points, k=k + 1)
    neighbors = points[idxs]

    # 1. 计算质心和协方差
    centroids = np.mean(neighbors, axis=1)  # (N, 3)
    centered = neighbors - centroids[:, np.newaxis, :]
    cov = np.einsum('nij,nik->njk', centered, centered) / k

    # 2. 计算特征值和特征向量
    # eigh 返回升序特征值和对应的特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 3. 基础曲率 (Surface Variation)
    lambda_sum = np.sum(eigenvalues, axis=1)
    curvature = eigenvalues[:, 0] / (lambda_sum + 1e-8)
    curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-6)
    # 4. 确定符号 (Direction check)
    # 最小特征值对应的特征向量即为该点的法向量 n
    normals = eigenvectors[:, :, 0]  # (N, 3)

    # 计算点到邻域质心的向量 v
    # v = P_self - P_centroid
    v = points - centroids

    # 计算 v 和 n 的点积。
    # 如果点积为正，说明点在邻域平面的“外侧”（凸）；为负则在“内侧”（凹）
    # 注意：这取决于法线 n 的朝向（向内还是向外）
    sign = np.sign(np.einsum('ni,ni->n', v, normals))

    return curvature * sign


def augment_non_uniform_scaling(p_points, landmarks, p_normals, p_curv, scale_range=(0.8, 1.2)):
    # 1. 产生三个轴独立的随机缩放因子
    scales = np.random.uniform(scale_range[0], scale_range[1], size=3)

    # 2. 坐标点缩放: P' = P * S
    p_points_aug = p_points * scales

    landmarks_aug = None
    if len(landmarks) >= 1:
        landmarks_aug = landmarks * scales

    # 3. 法线缩放: N' = N * (1/S)
    inv_scales = 1.0 / (scales + 1e-8)
    p_normals_aug = p_normals * inv_scales

    # 4. 必须重新归一化法线
    norm = np.linalg.norm(p_normals_aug, axis=-1, keepdims=True)
    p_normals_aug = p_normals_aug / (norm + 1e-8)

    # 5. 曲率保持不变
    p_curv_aug = p_curv

    return p_points_aug, landmarks_aug, p_normals_aug, p_curv_aug