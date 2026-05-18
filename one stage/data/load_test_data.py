
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

def augment_non_uniform_scaling(p_points, teeth_landmarks, scale_range=(0.8, 1.2)):
    # 1. 产生三个轴独立的随机缩放因子
    scales = np.random.uniform(scale_range[0], scale_range[1], size=3)

    # 2. 坐标点缩放: P' = P * S
    p_points_aug = p_points * scales

    for i, clss_ in enumerate(teeth_landmarks):
        landmarks = np.array(teeth_landmarks[clss_])

        landmarks_aug = landmarks * scales
        teeth_landmarks[clss_] = landmarks_aug

    return p_points_aug, teeth_landmarks

def random_axis_rotate(data_points, teeth_landmarks):

    v1 = np.sign(np.random.normal(0, 1, size=(1))[0])
    rotaxis = np.random.random(3) * 2 - 1 + 0.01
    rotaxis = rotaxis / np.linalg.norm(rotaxis)

    angle_ = v1 * cfg.Angles[np.random.randint(0, len(cfg.Angles), 1)[0]]  # [-10°--10°]
    rt = rotate_maxtrix(rotaxis, angle_)
    rt = rt[0:3, 0:3]

    data_points = (rt.dot(data_points.T)).T

    for i, clss_ in enumerate(teeth_landmarks):
        landmarks = np.array(teeth_landmarks[clss_])
        landmarks = (rt.dot(landmarks.T)).T
        teeth_landmarks[clss_] = landmarks

    return data_points, teeth_landmarks


def st_random_axis_trans(data_points, landmarks, scalev=1.0):
    trans_v = np.array([[-1, -1, 1]]) * scalev

    v1 = np.random.normal(0, 1, size=(1))[0]
    v2 = np.random.normal(0, 1, size=(1))[0]
    v3 = np.random.normal(0, 1, size=(1))[0]
    fg = np.clip(np.array([[v1, v2, v3]]), -1, 1)
    data_points = data_points + fg * trans_v
    landmarks = landmarks + fg * trans_v

    return data_points, landmarks



def read_data(file_path):


    file_data = np.load(file_path, allow_pickle=True).item()

    se_index = file_data["index"]
    mesh_points = file_data["points"]
    landmarks = file_data["landmarks"]
    seg_label = file_data["segment"]

    # if mesh_points.shape[0] > cfg.sam_points:
    #     se_index = np.random.choice(mesh_points.shape[0], size=cfg.sam_points, replace=False)
    # else:
    #     se_index = np.random.randint(0, mesh_points.shape[0], cfg.sam_points)

    data_points = mesh_points[se_index]
    seg_label = seg_label



    return data_points, landmarks, seg_label







class TestData():
    def __init__(self, file_root):
        self.data_dir = file_root

        self.train_list = None
        self.prepare(self.data_dir)

    def prepare(self, file_path):

        file_list = []
        get_files(file_path, file_list, "_c.npy")

        self.train_list = file_list

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, item):

        file_path = self.train_list[item]

        data_points, teeth_landmarks, seg_label = read_data(file_path)

        #Decentering data
        cpoint = np.mean(data_points.reshape(-1, 3), axis=0, keepdims=True)
        data_points = data_points - cpoint.reshape(1, 3)


        for i, clss_ in enumerate(teeth_landmarks):
            if len(teeth_landmarks[clss_]) >= 1:
                teeth_landmarks[clss_] = np.array(teeth_landmarks[clss_]) - cpoint


        #random axis rotate
        #data_points, teeth_landmarks = random_axis_rotate(data_points, teeth_landmarks)

        # random  scaling
        #data_points, teeth_landmarks = augment_non_uniform_scaling(data_points, teeth_landmarks, scale_range=(0.8, 1.2))

        #heat_map, offest_map = get_hotmap(data_points, teeth_landmarks)
        #pcurvature = get_calculate_curvature(data_points, teeth_landmarks, tooth_point_mask)


        data_points = torch.tensor(data_points)
        #heat_map = torch.tensor(heat_map)
        #offest_map = torch.tensor(offest_map)


        return data_points, teeth_landmarks, file_path

def get_hotmap(points, teeth_landmarks):


    heat_map = np.zeros((cfg.sam_points, cfg.landmarks_class), np.float32)
    offest_map = np.zeros((cfg.sam_points, cfg.landmarks_class, 3), np.float32)
    knums =20
    for i, clss_ in enumerate(teeth_landmarks):
        landmarks = np.array(teeth_landmarks[clss_])

        indices, distances = find_nearest_k_points(points, landmarks, k=knums)

        offv = points[indices].reshape(landmarks.shape[0], knums, 3) - landmarks.reshape(landmarks.shape[0], 1, 3)
        offv = offv.reshape(landmarks.shape[0]*knums, 3)


        sort_idx = np.argsort(distances)
        indices_sorted = indices[sort_idx]
        distances_sorted = distances[sort_idx]
        offv_sorted = offv[sort_idx]

        # 2. 找到每个索引第一次出现的位置（即最小距离所在位置）
        _, unique_first_indices = np.unique(indices_sorted, return_index=True)

        # 3. 提取不重复索引及其对应的最小距离
        final_indices = indices_sorted[unique_first_indices]
        final_distances = distances_sorted[unique_first_indices]
        final_offv = offv_sorted[unique_first_indices]

        sigma = 0.7
        gaussian_weights = np.exp(-(final_distances ** 2) / (2 * sigma ** 2))
        gaussian_weights = np.sort(gaussian_weights)
        # 4. 填充 heat_map
        heat_map[final_indices, i] = gaussian_weights
        offest_map[final_indices, i, :] = final_offv


    return heat_map, offest_map

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


def get_calculate_curvature(data_points, teeth_landmarks, tooth_point_mask):

    mask = np.sum(tooth_point_mask, axis=-1)
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


