import os
import torch
import trimesh
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from data.load_test_data import get_files
from data.data_process_oneheat import FurthestPointSampling,color_space
from config import config as cfg

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
def project_keypoints_to_mesh_surface(mesh, keypoints):
    """
    将带有置信度的关键点投影到 mesh 的几何表面上。

    参数:
        mesh: trimesh.Trimesh 对象
        keypoints: numpy.ndarray, 维度为 (6, 50, 4)，格式为 [x, y, z, conf]

    返回:
        projected_keypoints: numpy.ndarray, 维度为 (6, 50, 4)，坐标已更新为表面坐标
    """
    # 1. 分离坐标 (x, y, z) 和 置信度 (conf)
    # xyz 维度: (6, 50, 3), conf 维度: (6, 50, 1)
    xyz = keypoints[..., :3]
    conf = keypoints[..., 3:]

    # 2. 将坐标展平为二维数组 (N, 3) 以匹配 trimesh API 要求
    # 展平后维度变为 (300, 3)
    xyz_flattened = xyz.reshape(-1, 3)

    # 3. 使用 trimesh 的 proximity 查询点到表面的确切投影
    # mesh.nearest.on_surface 会计算到三角面的最近投影点，而不是最近顶点
    # 返回值包含:
    # closest_points: 投影在面上的坐标 (N, 3)
    # distances: 原始点到投影点的距离 (N,)
    # triangle_id: 点所在的三角面片索引 (N,)
    closest_points, distances, triangle_id = mesh.nearest.on_surface(xyz_flattened)

    # 4. 将计算出的表面坐标恢复到原始的高维形状 (6, 50, 3)
    xyz_projected = closest_points.reshape(xyz.shape)

    # 5. 将新坐标与原有的置信度重新拼接成 (6, 50, 4)
    projected_keypoints = np.concatenate([xyz_projected, conf], axis=-1)

    return projected_keypoints


def visualize_mesh_with_landmarks(tri_mesh, tooth_info, heatmap=None):
    # 1. 初始化 PyVista 渲染器
    plotter = pv.Plotter(window_size=[784, 784])

    # 2. 将 trimesh 对象直接转换为 pyvista 对象
    pv_mesh = pv.wrap(tri_mesh)

    # ==========================================
    # 核心修改：渲染连续热图
    # ==========================================
    if heatmap is not None:
        # 确保 heatmap 是 numpy 数组，且展平为一维
        heatmap_array = np.array(heatmap).flatten()

        # 检查 heatmap 长度是否与顶点数一致
        if len(heatmap_array) != pv_mesh.n_points:
            print(f"警告：热图长度 ({len(heatmap_array)}) 与顶点数量 ({pv_mesh.n_points}) 不一致！")

        # 将热图数据绑定到网格的 point_data
        pv_mesh.point_data["Heatmap"] = heatmap_array

        # 使用 scalars="Heatmap" 激活热图渲染
        # cmap: 色带，推荐 'jet' (蓝->绿->红) 或 'viridis'，'Reds'
        # clim: 强制将颜色映射范围固定在 0.0 到 1.0 之间
        plotter.add_mesh(pv_mesh,
                         scalars="Heatmap",
                         cmap="jet",
                         clim=[0.0, 1.0],
                         opacity=1.0,
                         smooth_shading=True,
                         show_scalar_bar=True)  # 可以设置为 False 隐藏颜色条
    else:
        # 如果没有传入热图，依然使用单一颜色渲染
        plotter.add_mesh(pv_mesh, color='#D3D3D3', opacity=1.0, smooth_shading=True, label="Teeth Mesh")
    # ==========================================

    # 3. 遍历并添加关键点球体 (保持你原有的逻辑)
    for i, tid in enumerate(tooth_info):
        tooth_kp = tooth_info[tid]

        kp_pos = np.array(tooth_kp).reshape(-1, 3)

        # 获取颜色：注意 PyVista 的 RGB 列表通常接受 0.0 到 1.0 的浮点数
        point_color = color_space.get(tid, [255, 255, 255])
        pv_color = [c / 255.0 for c in point_color]

        for ki in range(kp_pos.shape[0]):
            # 直接在指定中心点生成球体
            sphere = pv.Sphere(radius=0.8, center=kp_pos[ki])

            # 将球体加入渲染器并赋色
            plotter.add_mesh(sphere, color=pv_color)

    # 4. 显示场景
    plotter.view_xy()
    plotter.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def find_nearest_k_points(points, landmarks, k=5):
    tree = KDTree(points)
    distances, indices = tree.query(landmarks, k=k)
    return indices, distances


import torch
import numpy as np


def _process_weighted_logic(heatmap, offset_raw, raw_pos, idx, topk_indices, dist_thresh=1.0):
    """
    内部核心逻辑：根据给定的 topk_indices 计算加权中心和置信度
    heatmap: (B, N, C_sub)
    offset_raw: (B, N, C_sub, 3)
    topk_indices: (B, C_sub, K_top)
    """
    B, N, C_sub = heatmap.shape
    _, _, K_top = topk_indices.shape
    device = heatmap.device

    B_idx = torch.arange(B).view(-1, 1, 1).to(device)
    C_idx = torch.arange(C_sub).view(1, -1, 1).to(device)

    # 1. 提取种子点信息
    cand_raw_pos = raw_pos[B_idx, topk_indices]  # (B, C_sub, K_top, 3)
    conf_scores = heatmap[B_idx, topk_indices, C_idx]  # (B, C_sub, K_top)
    seed_neighbor_idx = idx[B_idx, topk_indices]  # (B, C_sub, K_top, K_knn)

    # 2. 提取邻域信息
    B_idx_n = torch.arange(B).view(-1, 1, 1, 1).to(device)
    C_idx_n = torch.arange(C_sub).view(1, -1, 1, 1).to(device)

    neighbor_raw_pos = raw_pos[B_idx_n, seed_neighbor_idx]
    neighbor_offset = offset_raw.detach()[B_idx_n, seed_neighbor_idx, C_idx_n]
    neighbor_heat = heatmap[B_idx_n, seed_neighbor_idx, C_idx_n]

    # 3. 距离约束与加权中心
    neighbor_shifted_pos = neighbor_raw_pos + neighbor_offset
    dists_to_seed = torch.norm(neighbor_raw_pos - cand_raw_pos.unsqueeze(3), dim=-1)
    dist_mask = (dists_to_seed < dist_thresh).float()

    weights = neighbor_heat * dist_mask
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    weighted_pos = (neighbor_shifted_pos * weights.unsqueeze(-1)).sum(dim=3) / weights_sum

    # 4. 拼接 [x, y, z, score]
    result = torch.cat([weighted_pos, conf_scores.unsqueeze(-1)], dim=-1)
    return result


def get_standard_centers(heatmap, offset_raw, raw_pos, idx):
    """处理前 C-1 个单关键点通道"""
    B, N, C = heatmap.shape
    # 局部极大值筛选
    B_idx_all = torch.arange(B).view(-1, 1, 1).to(heatmap.device)
    neighbor_heat_all = heatmap[B_idx_all, idx, :]
    local_max = heatmap >= neighbor_heat_all.max(dim=2)[0]

    priority_map = torch.where(local_max, 2.0 + heatmap, 1.0 - heatmap)
    _, topk_indices = torch.topk(priority_map, k=1, dim=1)
    topk_indices = topk_indices.transpose(1, 2)  # (B, C, 1)

    return _process_weighted_logic(heatmap, offset_raw, raw_pos, idx, topk_indices)


def get_multi_peak_centers(heatmap, offset_raw, raw_pos, idx, max_peaks=6):
    """处理最后一个通道（牙尖点），提取最多 6 个局部极大值"""
    # 此时传入的 heatmap 应该是 (B, N, 1)
    B, N, _ = heatmap.shape

    # 局部极大值筛选
    B_idx_all = torch.arange(B).view(-1, 1, 1).to(heatmap.device)
    neighbor_heat_all = heatmap[B_idx_all, idx, :]
    local_max = heatmap >= neighbor_heat_all.max(dim=2)[0]

    priority_map = torch.where(local_max, 2.0 + heatmap, 1.0 - heatmap)
    # 在同一个通道内取前 6 个
    _, topk_indices = torch.topk(priority_map, k=max_peaks, dim=1)
    topk_indices = topk_indices.transpose(1, 2)  # (B, 1, 6)

    return _process_weighted_logic(heatmap, offset_raw, raw_pos, idx, topk_indices)


def get_weighted_centers(twoheatmap, twooffset_raw, local_pos, k=20, topk=1, dist_thresh=1.0):
    """总封装函数"""
    B, N, C = twoheatmap.shape
    device = twoheatmap.device

    # 1. 统一计算一次 KNN 索引
    dist_mat = torch.cdist(local_pos, local_pos)
    _, idx = torch.topk(dist_mat, k=k, largest=False, dim=-1)

    # 2. 处理前 C-1 个通道 (每个通道 1 个点)
    # result_std 形状: (B, C-1, 4)
    result_std = get_standard_centers(
        twoheatmap[:, :, :-1],
        twooffset_raw[:, :, :-1, :],
        local_pos,
        idx
    )
    maxv = torch.max(twoheatmap[:, :, -1:])
    # 3. 处理最后一个通道 (牙尖点，取 6 个点)
    # result_cusp 形状: (B, 1, 6, 4) -> 压平后 (B, 6, 4)
    result_cusp = get_multi_peak_centers(
        twoheatmap[:, :, -1:],
        twooffset_raw[:, :, -1:, :],
        local_pos,
        idx,
        max_peaks=6
    ).squeeze(1)

    # 4. 转换为 Numpy 并返回
    # 你可以根据需要决定是否在这里拼接，或者分开返回
    return {
        "fixed": result_std.detach().cpu().numpy(),  # (B, C-1, 4)
        "cusps": result_cusp.detach().cpu().numpy()  # (B, 6, 4)
    }








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

if __name__ =="__main__":
    file_path = "H:/teethMICCAI2022/train_land_onenet_data/Crowded_testc/"
    mesh_root = "H:/teethMICCAI2022/raw_data/test/data_part_7/"
    file_list = []
    get_files(file_path, file_list, "_c.npy")

    from model.tooth_landmarks_onet_oneheat_reg import TwoStageToothPipeline
    model = TwoStageToothPipeline()#
    model_path = "./outputs/seg_land_oneheat_final.pth"
    model_initial(model, model_path)
    model.cuda().eval().float()

    #model = torch.jit.load('./save_model/seg_land_oneheat_final.pt').cuda().eval().float()
    for i in range(0, len(file_list)):#Crowded_testc33,, Missing Teeth 2
        file_path = file_list[i]
        print(i, "  ", file_path)
        data_name = os.path.basename(file_path).replace("_c.npy", "")
        arch_type = data_name.split("_")[-1]

        file_mesh_path = os.path.join(mesh_root, arch_type) + "/" +data_name.split("_")[0] + "/" + data_name+ ".obj"


        mesh = trimesh.load(file_mesh_path)
        vertics = mesh.vertices
        if vertics.shape[0]>8192*2:
            index = FurthestPointSampling(vertics.astype(np.float32), npoint=8192 * 2)
        else:
            index = np.random.randint(0, vertics.shape[0], 8192*2)
        points = vertics[index]
        cpoint = np.mean(vertics.reshape(-1, 3), axis=0, keepdims=True)


        file_data = np.load(file_path, allow_pickle=True).item()
        landmarks = file_data["landmarks"]


        with torch.no_grad():
            input_points = torch.tensor(points - cpoint).unsqueeze(dim=0).cuda().float()
            preheat_map, preoff_map, cls, init_land, init_conf, shifted_pos_all,twoheatmap, twooffset_raw, twoseg_mask, final_centers, relative_pos = model(input_points)
        key_points = shifted_pos_all.squeeze().detach().cpu().numpy()
        np.savetxt("./outputs/key_points"  + ".txt", key_points+cpoint, fmt='%.6f %.6f %.6f')
        probabilities = preheat_map.squeeze().detach().cpu().numpy()
        colored_points = assign_colors_by_probability(points, probabilities)
        save_colored_points("./outputs/heat_points"  + ".txt", colored_points)

        local_pos = final_centers.squeeze(dim=0).unsqueeze(dim=1)+ relative_pos
        pre_points = get_weighted_centers(twoheatmap, twooffset_raw, local_pos, k=20, topk=1, dist_thresh=1.0)



        final_centers = final_centers.squeeze().detach().cpu().numpy()+ cpoint.reshape(1, 3)
        init_conf = init_conf.squeeze().detach().cpu().numpy()
        init_land = init_land.squeeze(dim=0).detach().cpu().numpy().reshape(20, 1, 3)
        shifted_pos_all = shifted_pos_all.permute(1, 0, 2).detach().cpu().numpy()
        init_land[..., :3] = init_land[..., :3] + shifted_pos_all + cpoint.reshape(1, 1, 3)

        # input_data = points - cpoint
        # ptcp = shifted_pos_all[init_conf>0.5].reshape(-1, 3)
        # indices, distances = find_nearest_k_points(input_data, ptcp, k=1536)
        #
        # tooth_point = input_data[indices]
        # for ti in range(tooth_point.shape[0]):
        #     file_ = open("./outputs/tooth_point"+str(ti)+".txt", "w")
        #     tpoints = tooth_point[ti]
        #     for i in range(tpoints.shape[0]):
        #         point = tpoints[i]
        #         file_.write(str(point[0])+" " + str(point[1])+" " +str(point[2])+" \n")
        # file_.close()


        tooth_data = {}
        for i in  range(init_conf.shape[0]):
            tconf = init_conf[i]
            kp = init_land[i]

            if tconf>0.1 and len(kp)>=1:
                tooth_data[i] = kp
        hotmap = np.zeros((mesh.vertices.shape[0]), np.float32)
        hotmap[index] = probabilities


        tooth_data = {}
        for i in  range(final_centers.shape[0]):
            tooth_data[i] = final_centers[i]

        fixed = pre_points["fixed"]
        cusps = pre_points["cusps"]

        tooth_data = {}
        for i in  range(fixed.shape[0]):

            fpoints, fconf = fixed[i][...,:3].reshape(-1, 3), fixed[i][...,3:].reshape(-1)
            mpoints, mconf = cusps[i][...,:3].reshape(-1, 3), cusps[i][...,3:].reshape(-1)
            kep = np.concatenate([fpoints[:-1], mpoints], axis=0)
            conf = np.concatenate([fconf[:-1], mconf], axis=0)
            kep = kep[conf >0.2]


            tooth_data[i] = kep + cpoint.reshape(1, 3)

        visualize_mesh_with_landmarks(mesh, tooth_data,heatmap=None)

