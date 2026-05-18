import os
import json
import random
import trimesh
import numpy as np
import torch
from data.util import walkFileType, walkFile, data_ori_align, sindata_rot_trans


kp_color_map = {
    "Mesial": [0, 0, 139, 255], "Distal": [135, 206, 235, 255],
    "Cusp": [255, 255, 0, 255], "InnerPoint": [255, 255, 255, 255],
    "OuterPoint": [255, 182, 193, 255], "FacialPoint": [255, 165, 0, 255]
}

tooth_key = { 1: ['Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
              2: ['Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
              3: ['Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
              4: ['Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
              5: ['Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
              6: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
              7: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
              8: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
              9: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
             10: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
             11: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
             12: ['Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
             13: ['Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
             14: ['Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
             15: ['Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
             16: ['Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint', 'FacialPoint']
     }




def get_tooth_color(tooth_id):
    # 使用牙齿编号作为随机种子，保证每次运行颜色一致
    random.seed(int(tooth_id))
    return [random.randint(50, 255) for _ in range(3)] + [255]


def get_files(file_dir, file_list, type_str):

    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(path, file_list, type_str)
        else:
            if file_.rfind(type_str) !=-1:
                file_list.append(path)



def data_process():
    file_path = "I:/mesh_tooth_data/pred_landmarks_data/train_data/"
    file_list = []
    get_files(file_path, file_list, "npy")

    data_points= np.zeros((16, len(file_list)), np.int32)
    for di in range(len(file_list)):



        file_data = np.load(file_list[di], allow_pickle=True).item()

        teeth_nums = []
        teeth_points = []
        for key in file_data:
            teeth_nums.append(int(key))
            landmarks = file_data[key][0]
            teeth_points.append(np.array(file_data[key][1]))
            data_points[int(key)-1, di] = len(landmarks)
        teeth_nums = np.array(teeth_nums)
        order_index = np.argsort(teeth_nums)
        teeth_nums = teeth_nums[order_index]
    print(data_points)

    for i in  range(data_points.shape[0]):
        print(set(data_points[i].tolist()))


def read_data(file_list):

    tooth_key = {}
    tooth_data ={}

    for ti in range(len(file_list)):
        tid = os.path.basename(file_list[ti]).replace(".obj", "")
        landmark_path = file_list[ti].replace("obj", "json")
        if os.path.exists(landmark_path):

            mesh = trimesh.load_mesh(file_list[ti])
            with open(landmark_path, 'r', encoding='utf-8') as f:
                feat_data = json.load(f)
                key_point = {}
                for i in range(len(feat_data)):
                    key_point[i] = [feat_data[i]["class"], np.array(feat_data[i]["coord"])]

            tid = 9-int(tid) if int(tid) <=8 else int(tid)
            tooth_data[int(tid)] = [int(tid), key_point, mesh]
            tooth_key[int(tid)] = key_point.keys()


    print(tooth_key)
    return [tooth_data]





def toothMICCAI2022Data():
    data_points = np.load("tooth_data.npy")
    #
    print(np.max(data_points, axis=1))


    data_path = "H:/teethMICCAI2022/TLDETR_data/"

    dir_list = []
    walkFile(data_path, dir_list)

    data_points = np.zeros((16, len(dir_list)), np.int32)
    save_root = "H:/teethMICCAI2022/TLDETR_data_train/train/"
    for di in range(90, len(dir_list), 1):
        print(di, " ", dir_list[di])
        folder_name = os.path.basename(dir_list[di])

        file_list = []
        get_files(dir_list[di], file_list, "obj")
        tooth_data = read_data(file_list)

        lower_data = tooth_data[0]

        rot_matrix = data_ori_align(lower_data)
        tooth_data, tooth_data_points, matrix = sindata_rot_trans(lower_data, rot_matrix)


        np.save(save_root + folder_name + "_cv.npy", tooth_data_points)
        # for ki, tid in enumerate(tooth_data):
        #     tid, key_point, mesh = tooth_data[tid]
        #     data_points[tid-1, di] = len(key_point)
    # np.save("tooth_data.npy", data_points)

    #print("")
        # tooth_data = [tooth_data]
        # for type_, t_data in enumerate(tooth_data):
        #     visual_elements = []
        #     for i, tid in enumerate(t_data):
        #         tid, tooth_kp, mesh = t_data[tid]
        #
        #         visual_elements.append(mesh)
        #         current_tooth_color = get_tooth_color(tid)
        #         # 遍历该牙齿的所有关键点
        #         # if "WALA" in tooth_kp:
        #         #     del tooth_kp["WALA"]
        #
        #         for kp_i, kp_pos in tooth_kp.items():
        #             kp_name = kp_pos[0]
        #             kp_pos = kp_pos[1]
        #             # 提取缩写，防止 kp_name 是 "MCP_1" 这种格式
        #             short_name = kp_name.split("_")[0]
        #
        #             # 获取预定义颜色，如果没找到则默认使用白色
        #             point_color = kp_color_map.get(short_name, [255, 255, 255, 255])
        #
        #             sphere = trimesh.creation.uv_sphere(radius=0.5)
        #             sphere.visual.face_colors = point_color
        #
        #             # 平移并添加到 visual_elements
        #             translation = np.eye(4)
        #             translation[:3, 3] = kp_pos
        #             sphere.apply_transform(translation)
        #             visual_elements.append(sphere)
        #
        #             # 3. 打印调试信息（因为 trimesh 默认 viewer 很难直接在 3D 空间显示文字标签）
        #             #print(f"Tooth {tid} - Landmark: {kp_name} at {kp_pos}")
        #
        #         # 创建场景并显示
        #     scene = trimesh.Scene(visual_elements)
        #     scene.camera.perspective = False
        #
        #     # 2. 设置正面视角
        #     # 根据你的截图，牙齿的正面通常对应 (np.pi/2, 0, 0) 或者 (0, 0, 0)
        #     # 我们使用 set_camera 自动计算距离，确保模型填满窗口
        #     scene.set_camera(angles=(np.pi / 2, 0, 0), distance=100, center=scene.centroid-10)
        #
        #     scene.show()  # 这会打开一个交互式窗口


def rand_select_train_data():
    import random
    import shutil
    file_path = "E:/DataSet/mesh_tooth_landmark_data/Teeth3DS_data_train/train/"
    file_list = []
    get_files(file_path, file_list, ".npy")
    random.shuffle(file_list)

    save_root = "E:/DataSet/mesh_tooth_landmark_data/Teeth3DS_data_train/test/"
    for i in range(40):

        file_data = file_list[i]
        end_name = os.path.split(file_data)[-1]
        dst_path = save_root + end_name


        shutil.move(file_data, dst_path)


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


def farthest_point_sample_cuda(
        points: torch.Tensor,
        K: int,
        random_start: bool = True
) -> torch.Tensor:
    """
    PyTorch CUDA 版最远点采样（FPS），适配 (N, 3) 点云，返回采样索引
    核心：利用 CUDA 张量的向量化运算，全程在 GPU 上执行，无CPU/GPU数据拷贝

    Args:
        points: 点云张量，形状 (N, 3)，必须是 CUDA 张量（torch.float32/torch.float64）
        K: 采样点数，满足 0 < K ≤ N
        random_start: 是否随机选择起始点

    Returns:
        sampled_indices: 采样索引张量，形状 (K,)，CUDA 张量（torch.int32）

    Raises:
        ValueError: 输入形状/设备/参数不合法
    """
    # 1. 严格校验输入
    if points.device.type != "cuda":
        raise ValueError(f"点云必须是 CUDA 张量，当前设备：{points.device}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"点云形状必须为 (N, 3)，当前：{points.shape}")

    N = points.shape[0]
    if K > N or K <= 0:
        raise ValueError(f"采样数K={K} 必须满足 0 < K ≤ {N}")

    # 2. 初始化
    device = points.device
    sampled_indices = torch.zeros(K, dtype=torch.int32, device=device)
    min_distances = torch.full((N,), float("inf"), device=device)  # 每个点到已选点集的最小距离

    # 3. 选择第一个点
    if random_start:
        start_idx = torch.randint(0, N, (1,), device=device, dtype=torch.int32)
    else:
        start_idx = torch.tensor([0], dtype=torch.int32, device=device)
    sampled_indices[0] = start_idx

    # 4. 核心迭代（CUDA 加速）
    for k in range(1, K):
        # 获取上一轮选中的点 (3,)
        last_selected = points[sampled_indices[k - 1]]
        # 向量化计算所有点到上一轮选点的距离平方（CUDA 并行计算）
        dist = torch.sum((points - last_selected) ** 2, dim=1)
        # 更新最小距离
        min_distances = torch.min(min_distances, dist)
        # 选择距离最大的点（CUDA 并行查找）
        sampled_indices[k] = torch.argmax(min_distances)

    return sampled_indices.detach().cpu().numpy()


def farthestPointSampling():

    save_root = "E:/DataSet/mesh_tooth_landmark_data/seg_landmarks_train_data/train_farthest/"

    file_path ="E:/DataSet/mesh_tooth_landmark_data/seg_landmarks_train_data/train/"
    file_list = []
    get_files(file_path, file_list, ".npy")
    for i in range(0, len(file_list)):

        file_path = file_list[i]
        data_name = os.path.basename(file_path).replace("_cv1.npy", "")

        file_data = np.load(file_path, allow_pickle=True).item()

        for key in file_data:
            landmarks_ = file_data[key]["key_point"]
            mesh_points = np.array(file_data[key]["points"])[..., :3]
            mesh_points = mesh_points.astype(np.float32)
            #mesh_points = torch.tensor(mesh_points).cuda().float()
            if mesh_points.shape[0]>=256:
                sindices1 = farthest_point_sample_np(points=mesh_points,K=256, random_start=True)
            else:
                sindices1 = np.random.randint(0, mesh_points.shape[0], 256)

            if mesh_points.shape[0]>=512:
                sindices2 = farthest_point_sample_np(points=mesh_points,K=512, random_start=True)
            else:
                sindices2 = np.random.randint(0, mesh_points.shape[0], 512)

            if mesh_points.shape[0]>=1024:
                sindices3 = farthest_point_sample_np(points=mesh_points,K=1024, random_start=True)
            else:
                sindices3 = np.random.randint(0, mesh_points.shape[0], 1024)

            if mesh_points.shape[0]>=2048:
                sindices4 = farthest_point_sample_np(points=mesh_points,K=2048, random_start=True)
            else:
                sindices4 = np.random.randint(0, mesh_points.shape[0], 2048)

            if mesh_points.shape[0]>=4096:
                sindices5 = farthest_point_sample_np(points=mesh_points,K=4096, random_start=True)
            else:
                sindices5 = np.random.randint(0, mesh_points.shape[0], 4096)

            file_data[key]["farthestindex"] = [sindices1, sindices2, sindices3,  sindices4, sindices5]

        save_path = save_root + data_name + "_fa.npy"
        np.save(save_path, file_data)
        print(i,  "     ", file_path)

if __name__ == "__main__":

    #data_process()
    #toothMICCAI2022Data()
    #rand_select_train_data()

    #farthestPointSampling()


    print("")
    # all_files =[]
    # file_path = "F:/teethMICCAI2022/3DTeethLand_landmarks_train/"
    # get_files(file_path, all_files, "__kpt.json")
    #
    # file_path ="E:/DataSet/mesh_tooth_landmark_data/seg_landmarks_train_data/train/"
    # file_list = []
    # get_files(file_path, file_list, ".npy")
    # import shutil
    # for i in range(len(all_files)):
    #     data_name = os.path.basename(all_files[i]).replace("__kpt.json", "_cv1.npy")
    #     src_path = "E:/DataSet/mesh_tooth_landmark_data/seg_landmarks_train_data/train/"
    #     dst_path = "E:/DataSet/mesh_tooth_landmark_data/seg_landmarks_train_data/seg_land/"
    #
    #     shutil.copy(src_path + data_name, dst_path+data_name)

