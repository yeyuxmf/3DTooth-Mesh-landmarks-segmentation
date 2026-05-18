
import os
import copy
import torch
import numpy as np
from data.util import rotate_maxtrix
from config import config as cfg
from scipy.spatial import cKDTree

def get_files(file_dir, file_list, type_str):

    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(path, file_list, type_str)
        else:
            if file_.rfind(type_str) !=-1:
                file_list.append(path)


def read_data(file_path):


    file_data = np.load(file_path, allow_pickle=True).item()


    tooth_mask = np.zeros((cfg.tooth_nums, 2), np.int32)
    data_points = np.zeros((cfg.tooth_nums, cfg.sam_points, 3), np.float64)

    # random remove tooth
    for key in file_data:

        landmarks_ = file_data[key][0]
        mesh_points = np.array(file_data[key][1])
        se_index = np.random.randint(0, mesh_points.shape[0], cfg.sam_points)
        select_points = mesh_points[se_index]



        tooth_mask[int(key) - 1, 0] = 1
        if key< 6 or key >11:
            tooth_mask[int(key) - 1, 1] = 1

        data_points[int(key) - 1] = select_points
    teeth_landmarks = {'Mesial':[], 'Distal':[], 'InnerPoint':[], 'OuterPoint':[], 'FacialPoint':[], 'Cusp':[]}
    for i in range(len(landmarks_)):
        kclass_, coord =  landmarks_[i]
        teeth_landmarks[kclass_].append(coord)


    return data_points, teeth_landmarks, tooth_mask






class TestData():
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

        data_points, teeth_landmarks, tooth_point_mask = read_data(file_path)
        av = np.sum(tooth_point_mask)

        tooth_mask = tooth_point_mask[:, 0]


        #Decentering data
        cpoint = np.mean(data_points[tooth_mask >=1].reshape(-1, 3), axis=0, keepdims=True)
        data_points[tooth_mask >=1] = data_points[tooth_mask >=1] - cpoint.reshape(1, 1, 3)
        for i , key in enumerate(teeth_landmarks):
            teeth_landmarks[key] = np.array(teeth_landmarks[key])  - cpoint


        pcurvature = get_calculate_curvature(data_points, teeth_landmarks, tooth_point_mask)
        data_points = torch.tensor(np.concatenate([data_points, pcurvature], axis=-1))

        tooth_point_mask = torch.tensor(tooth_point_mask)


        return data_points, tooth_point_mask, teeth_landmarks

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