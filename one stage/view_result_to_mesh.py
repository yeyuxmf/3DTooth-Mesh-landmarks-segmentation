import os
import trimesh
import numpy as np
import pyvista as pv
import matplotlib.colors as mcolors
from data.load_test_data import get_files
from config import config as cfg


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





def visualize_mesh_with_landmarks(tri_mesh, landmarks_data, plotter, title, subplot_idx):

    #plotter = pv.Plotter(title="Teeth Landmarks Visualization")
    #plotter.set_background('white')
    #mesh = pv.wrap(tri_mesh)

    plotter.subplot(0, subplot_idx)
    plotter.add_mesh(tri_mesh, color='#D3D3D3', opacity=1.0, smooth_shading=True, label="Teeth Mesh")
    # 2. 定义 6 类醒目的基础颜色
    class_base_colors = [
        [1.0, 0.0, 0.0],  # 红
        [0.0, 0.8, 0.0],  # 绿
        [0.0, 0.0, 1.0],  # 蓝
        [1.0, 0.5, 0.0],  # 橙
        [0.6, 0.0, 0.8],  # 紫
        [0.0, 0.8, 0.8]  # 青
    ]

    # 3. 循环处理每一类关键点
    for class_idx in range(6):
        data = landmarks_data[class_idx]  # (50, 4)
        coords = data[:, :3]
        confs = np.clip(data[:, 3], 0.0, 1.0)  # 置信度 0-1
        mask = confs<0.3

        confs[mask] = np.power(confs[mask], 3)
        # --- 核心逻辑：手动构建 RGBA 矩阵 (50, 4) ---
        base_rgb = np.array(class_base_colors[class_idx])
        bg_rgb = np.array([1.0, 1.0, 1.0])  # 白色背景色

        # A. 计算颜色淡化 (RGB)
        # 颜色 = 基础色 * conf + 背景色 * (1 - conf)
        mixed_rgb = confs[:, None] * base_rgb + (1 - confs[:, None]) * bg_rgb

        # B. 组合成 RGBA
        # 最后一列是 Alpha，直接用 conf
        rgba_array = np.zeros((50, 4))
        rgba_array[:, :3] = mixed_rgb  # 前三列是颜色
        rgba_array[:, 3] = confs  # 第四列是透明度

        # 创建点云
        point_cloud = pv.PolyData(coords)
        # 将 RGBA 数组直接放入点云
        point_cloud["rgba_data"] = (rgba_array * 255).astype(np.uint8)

        # 4. 添加点云到场景
        plotter.add_mesh(
            point_cloud,
            scalars="rgba_data",
            rgba=True,  # 关键点：明确告诉它这是 RGBA 数据
            point_size=10.0,
            render_points_as_spheres=True,
            name=f"class_{class_idx}",
            show_scalar_bar=False,
            ambient=0.5
        )
        plotter.add_text(title, font_size=12)
        plotter.view_isometric()
    print("已修复广播错误。Mesh 100% 不透明，关键点根据概率褪色并变透明。")
    #plotter.show()




if __name__ =="__main__":
    file_path = "H:/teethMICCAI2022/train_land_onenet_data/Missing_testc/"
    mesh_root = "H:/teethMICCAI2022/raw_data/test/data_part_7/"
    file_list = []
    get_files(file_path, file_list, "_c.npy")


    for i in range(2, len(file_list)):#Crowded_testc33,, Missing Teeth 2
        file_path = file_list[i]
        print(i, "  ", file_path)
        data_name = os.path.basename(file_path).replace("_c.npy", "")
        arch_type = data_name.split("_")[-1]

        file_mesh_path = os.path.join(mesh_root, arch_type) + "/" +data_name.split("_")[0] + "/" + data_name+ ".obj"
        file_hot_path = file_path.replace("_c.npy", "_hotmap.npy")
        file_our_path = file_path.replace("_c.npy", "_ours.npy")

        mesh = trimesh.load(file_mesh_path)
        vertics = mesh.vertices
        cpoint = np.mean(vertics.reshape(-1, 3), axis=0, keepdims=True)

        file_data = np.load(file_path, allow_pickle=True).item()
        landmarks = file_data["landmarks"]

        real_landmark = np.zeros((6, 50, 4))
        for i, clss_ in enumerate(cfg.landPorder):
            if len(landmarks[clss_]) >= 1:
                rlandmark = np.array(landmarks[clss_]).reshape(-1, 3)
                rlandmark = np.concatenate([rlandmark, np.ones(rlandmark.shape[0]).reshape(-1, 1)], axis=-1)
                real_landmark[i, :rlandmark.shape[0], :] = rlandmark




        baseline_hot_result = np.load(file_hot_path)
        our_reg_result = np.load(file_our_path)
        baseline_hot_result[..., :3] = baseline_hot_result[..., :3] + cpoint
        our_reg_result[..., :3] = our_reg_result[..., :3] + cpoint

        #The prediction results of deep learning may deviate from the mesh surface,
        # so they need to be projected onto the surface.
        baseline_hot_result = project_keypoints_to_mesh_surface(mesh, baseline_hot_result)
        our_reg_result = project_keypoints_to_mesh_surface(mesh, our_reg_result)

        plotter = pv.Plotter(shape=(1, 3), window_size=[1500, 500])
        # plotter = pv.Plotter(title="Teeth Landmarks Visualization")
        # plotter.set_background('white')

        visualize_mesh_with_landmarks(mesh, baseline_hot_result, plotter, "Heatmap", subplot_idx=0)
        visualize_mesh_with_landmarks(mesh, our_reg_result, plotter, "Ours", subplot_idx=1)
        visualize_mesh_with_landmarks(mesh, real_landmark, plotter, "Ground Truth", subplot_idx=2)

        plotter.link_views()

        # 4. 最终显示
        plotter.show()
        print("over")