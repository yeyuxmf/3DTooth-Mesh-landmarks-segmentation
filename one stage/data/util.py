import os
import math
import random
import trimesh
import numpy as np



kp_color_map = {
    "MCP": [0, 0, 139, 255], "DCP": [135, 206, 235, 255],
    "CEP": [255, 255, 0, 255], "FA": [255, 255, 255, 255],
    "IEP": [0, 182, 193, 255], "DeCP": [255, 165, 0, 255],
    "BCP": [200, 0, 0, 255], "LCP": [199, 21, 133, 255],
    "MBCP": [255, 100, 100, 255], "DBCP": [139, 0, 0, 255],
    "MLCP": [218, 112, 214, 255], "DLCP": [128, 0, 128, 255],
    "CFP": [50, 205, 50, 255], "WALA": [169, 169, 169, 255],
    "Mesial": [0, 0, 139, 255], "Distal": [135, 206, 235, 255],
    "Cusp": [255, 255, 0, 255], "InnerPoint": [255, 255, 255, 255],
    "OuterPoint": [255, 182, 193, 255], "FacialPoint": [255, 165, 0, 255]
}

class AlignmentConfig:
    dim: int = 3                # 空间维度
    teeth_nums: int = 16        # 总牙齿数
    sam_points: int = 512       # 每颗牙齿采样的点数

    uAntTeeth = [7,8,9,10]
    urposTeeth = [12,13,14,15,16]
# 使用方式
config = AlignmentConfig()


def get_tooth_color(tooth_id):
    # 使用牙齿编号作为随机种子，保证每次运行颜色一致
    random.seed(int(tooth_id))
    return [random.randint(50, 255) for _ in range(3)] + [255]



def walkFile(path_root, file_list):
    # 列出路径下的所有项目
    for item in os.listdir(path_root):
        # 拼接成完整路径
        path_file = os.path.join(path_root, item)
        # 判断是否为目录
        if os.path.isdir(path_file):
            file_list.append(path_file)

def walkFileType(path_root, file_list, type_):

    for root, dirs, files in os.walk(path_root):
        # 遍历所有的文件夹
        for d in dirs:
            path_file = os.path.join(root, d)
            if type_ in path_file:
                file_list.append(path_file)
def get_files(file_dir, file_list, type_str):

    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(path, file_list, type_str)
        else:
            if file_.rfind(type_str) !=-1:
                file_list.append(path)

#############################################################################

def normalize_vectors(v):
    # 1. 计算范数 (模长)
    # ord=2 表示欧几里得距离，axis=-1 表示对最内层维度(XYZ)求模
    norm = np.linalg.norm(v, ord=2, axis=-1, keepdims=True)

    # 2. 避免除以 0 (加上一个极小值 eps)
    v_unit = v / (norm + 1e-9)
    return v_unit

def data_ori_align(teeth_data):

    all_points = []
    uAntTeeth =[]
    urposTeeth =[]
    for i, tid in enumerate(teeth_data):
        mesh_points = np.array(teeth_data[tid][2].vertices)
        if tid in config.uAntTeeth:
            uAntTeeth.append(mesh_points)
        if tid in config.urposTeeth:
            urposTeeth.append(mesh_points)
        all_points.append(mesh_points)
    urposTeeth = np.mean(np.concatenate(urposTeeth, axis=0), axis=0)
    uAntTeeth = np.mean(np.concatenate(uAntTeeth, axis=0), axis=0)
    all_points = np.concatenate(all_points, axis=0)

    cp = np.mean(all_points, axis=0)
    z_roi = normalize_vectors(uAntTeeth - cp)
    x_roi = normalize_vectors(urposTeeth - cp)
    y_roi = normalize_vectors(np.cross(z_roi, x_roi))
    x_roi = np.cross(y_roi, z_roi)

    rot = np.eye(4)
    rot[:3, :3] = np.stack([x_roi, y_roi, z_roi], axis=0)

    return  rot

def sindata_rot_trans(t_data, rot_matrix):

    t_points = []
    for i, tid in enumerate(t_data):
        tid, tooth_kp, mesh = t_data[tid]
        t_points.append(mesh.vertices)

    cp = np.mean(np.concatenate(t_points, axis=0), axis=0)

    translation_matrix = trimesh.transformations.translation_matrix(-cp)
    matrix = rot_matrix  @ translation_matrix
    tooth_data_points = {}
    tooth_data = {}
    for i, tid in enumerate(t_data):
        tid, key_point, mesh = t_data[tid]
        mesh.apply_transform(matrix)

        # for ki in key_point:
        #     key_point[ki][1] = np.dot(matrix[:3, :3], np.array(key_point[ki][1]))+ matrix[:3, 3]
        #
        # keys = [key_point[ki][0] for ki in key_point]
        # if tid <= 11 and tid >=6 and (len(key_point)>5 or "Cusp" in keys):
        #     continue

        tooth_data[tid] = [tid, key_point, mesh]
        tooth_data_points[int(tid)] = [key_point, mesh.vertices]
        #mesh.export("../outputs/" + str(tid) + "_upper.obj")

    return tooth_data, tooth_data_points, matrix

#############################################################################

def rotation_matrix(rotate_axis, rotate_angle):
    M_PI = math.pi
    axis = rotate_axis
    angle = rotate_angle

    m = np.zeros((4,4) ,np.float64)
    a = angle * (M_PI / 180.0)
    c = math.cos(a)
    s = math.sin(a)
    one_m_c = 1 - c
    ax = axis / np.sqrt(np.sum(np.power(axis, 2)))

    m[0, 0] = ax[0] * ax[0] * one_m_c + c
    m[0, 1] = ax[0] * ax[1] * one_m_c - ax[2] * s
    m[0, 2] = ax[0] * ax[2] * one_m_c + ax[1] * s

    m[1, 0] = ax[1] * ax[0] * one_m_c + ax[2] * s
    m[1, 1] = ax[1] * ax[1] * one_m_c + c
    m[1, 2] = ax[1] * ax[2] * one_m_c - ax[0] * s

    m[2, 0] = ax[2] * ax[0] * one_m_c - ax[1] * s
    m[2, 1] = ax[2] * ax[1] * one_m_c + ax[0] * s
    m[2, 2] = ax[2] * ax[2] * one_m_c + c

    m[3, 3] = 1.0

    return m

def rotate_maxtrix(rotaxis, angle_):
    M_PI = math.pi
    rt = np.eye(4)  #单位矩阵
    if (np.sqrt(rotaxis.dot(rotaxis)) > 0.001):
        rotaxis = rotaxis / np.sqrt(np.sum(np.power(rotaxis, 2)))
        rotangle = angle_
        rt = rotation_matrix(rotaxis, rotangle)

    return rt