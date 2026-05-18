
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

        if len(teeth_landmarks[clss_]) >= 1:

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
        if len(teeth_landmarks[clss_]) >= 1:

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
    #seg_label = seg_label[se_index]

    landmarks_ou = {}
    for lclss in cfg.landPorder:
        kep = landmarks[lclss]
        landmarks_ou[lclss] = kep


    return data_points, landmarks_ou, seg_label







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


        data_points, teeth_landmarks, seg_label = read_data(file_path)

        #Decentering data
        cpoint = np.mean(data_points.reshape(-1, 3), axis=0, keepdims=True)
        data_points = data_points - cpoint.reshape(1, 3)


        for i, clss_ in enumerate(teeth_landmarks):
            if len(teeth_landmarks[clss_])>=1:
                teeth_landmarks[clss_] = np.array(teeth_landmarks[clss_]) - cpoint


        #random axis rotate
        data_points, teeth_landmarks = random_axis_rotate(data_points, teeth_landmarks)

        # random  scaling
        data_points, teeth_landmarks = augment_non_uniform_scaling(data_points, teeth_landmarks, scale_range=(0.8, 1.2))

        heat_map, offest_map, class_map, mask = get_hotmap(data_points, teeth_landmarks)
        #pcurvature = get_calculate_curvature(data_points, teeth_landmarks, tooth_point_mask)

        # probabilities = np.max(heat_map, axis=-1)
        # colored_points = assign_colors_by_probability(data_points, probabilities)
        # save_colored_points("./outputs/heat_points.txt", colored_points)

        # file_ = open("./outputs/data_points.txt", "w")
        # for i in range(data_points.shape[0]):
        #     point = data_points[i]
        #     file_.write(str(point[0])+" " + str(point[1])+" " +str(point[2])+" \n")
        # file_.close()
        #
        # mask = heat_map >0.1
        # data_points = data_points.reshape(cfg.sam_points, 1, 3)
        # data_points = np.tile(data_points, (1, cfg.landmarks_class, 1))
        # dapoints = data_points[mask]  - offest_map[mask]
        #
        # file_ = open("./outputs/dapoints.txt", "w")
        # for i in range(dapoints.shape[0]):
        #     point = dapoints[i]
        #     file_.write(str(point[0])+" " + str(point[1])+" " +str(point[2])+" \n")
        # file_.close()




        data_points = torch.tensor(data_points)
        heat_map = torch.tensor(heat_map)
        mask = torch.tensor(mask)
        offest_map =torch.tensor(offest_map)
        class_map = torch.tensor(class_map)

        return data_points, heat_map, offest_map, class_map, mask


def get_hotmap(points, teeth_landmarks, sigma=0.7, knums=20):  #8192*1
#def get_hotmap(points, teeth_landmarks, sigma=1, knums=40):#8192*2
    num_pts = points.shape[0]
    num_classes = cfg.landmarks_class

    # 使用输入点的实际规模初始化
    heat_map = np.zeros((num_pts, num_classes-1), np.float32)
    offset_map = np.zeros((num_pts, num_classes-1, 3), np.float32)
    mask = np.zeros((num_pts, num_classes-1), np.uint8)
    class_map = np.zeros((num_pts, num_classes), np.uint8)
    for i, clss_ in enumerate(teeth_landmarks):
        landmarks_list = teeth_landmarks[clss_]
        if len(landmarks_list) < 1:
            continue

        landmarks = np.array(landmarks_list)  # (L, 3)

        # 1. 查找最近邻
        indices, distances = find_nearest_k_points(points, landmarks, k=knums)  # (L, K)

        # 2. 计算偏移向量 (Landmark - Point)
        # points[indices] 形状为 (L, K, 3)
        offv = landmarks[:, np.newaxis, :] - points[indices]

        # 3. 计算高斯权重 (不要做额外的 0-1 归一化)
        gaussian_weights = np.exp(-(distances ** 2) / (2 * (sigma ** 2)))

        # 4. 展平并处理冲突 (一个点对应多个 Landmark 时取权重最大的)
        flat_indices = indices.ravel()
        flat_weights = gaussian_weights.ravel()
        flat_offv = offv.reshape(-1, 3)

        # 按照权重从大到小排序
        sort_idx = np.argsort(-flat_weights)
        flat_indices = flat_indices[sort_idx]
        flat_weights = flat_weights[sort_idx]
        flat_offv = flat_offv[sort_idx]

        # 仅保留每个点第一次出现的索引（即最大权重）
        _, unique_first_indices = np.unique(flat_indices, return_index=True)

        final_indices = flat_indices[unique_first_indices]
        final_weights = flat_weights[unique_first_indices]
        final_offv = flat_offv[unique_first_indices]


        mask_w = flat_weights[unique_first_indices] > 0.5
        final_wei = final_weights[mask_w]
        final_indw = final_indices[mask_w]


        id = cfg.class_type[i]
        # 5. 填充结果
        heat_map[final_indw, id] = final_wei
        #heat_map[final_indices, id] = final_weights
        offset_map[final_indices, id, :] = final_offv
        mask[final_indices, id] = 1
        class_map[final_indices, i] = 1

    return heat_map, offset_map, class_map, mask


def find_nearest_k_points(points, landmarks, k=5):
    tree = KDTree(points)
    distances, indices = tree.query(landmarks, k=k)
    return indices, distances

def normalize_0_1(x, axis=-1):
    min_val = np.min(x, axis=axis, keepdims=True)
    max_val = np.max(x, axis=axis, keepdims=True)
    return (x - min_val) / (max_val - min_val + 1e-8)  # 避免除零

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