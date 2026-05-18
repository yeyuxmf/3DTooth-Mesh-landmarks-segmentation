import numpy as np
import pyvista as pv
from pathlib import Path


def load_custom_obj(file_path):
    """
    模块 1: 专用读取器
    手动解析 v x y z r g b 格式，确保暗色系颜色的准确提取
    """
    vertices = []
    colors = []
    faces = []

    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.split()
                    # 提取坐标
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    # 提取颜色 (r, g, b)
                    if len(parts) >= 7:
                        colors.append([int(float(parts[4])), int(float(parts[5])), int(float(parts[6]))])
                elif line.startswith('f '):
                    parts = line.split()
                    # OBJ 索引从 1 开始，转为从 0 开始
                    face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    faces.append([len(face)] + face)
    except Exception as e:
        print(f"文件读取出错 {file_path}: {e}")
        return None, None, None

    return np.array(vertices), np.array(faces), np.array(colors)


def visualize_with_synced_pose(mesh_list):
    """
    模块 2: 姿态同步可视化
    逐个显示网格，并保持上一个窗口关闭时的相机姿态。
    """
    if not mesh_list:
        print("没有可显示的数据。")
        return

    # 初始化相机位置变量
    current_camera_pos = None

    print(f"共加载 {len(mesh_list)} 个模型。")
    print(">>> 提示：在第一个窗口调好角度，关闭后下一个模型会保持该角度 <<<")

    for i, mesh_info in enumerate(mesh_list):
        file_name = mesh_info['name']
        v, f, c = mesh_info['v'], mesh_info['f'], mesh_info['c']

        print(f"正在展示 [{i + 1}/{len(mesh_list)}]: {file_name}")

        # 1. 创建 PyVista 网格
        mesh = pv.PolyData(v, f)

        # 2. 创建 Plotter
        plotter = pv.Plotter(title=f"牙齿同步视图 - {file_name}")
        plotter.background_color = "white"

        # 3. 添加带颜色的网格（保持之前的正确渲染设置）
        if len(c) > 0:
            mesh.point_data['LabelColors'] = c.astype(np.uint8)
            plotter.add_mesh(
                mesh,
                scalars='LabelColors',
                rgb=True,
                lighting=False,  # 关键：显示原始深色，不受阴影干扰
                interpolation='flat'  # 关键：看清分割边界
            )
        else:
            plotter.add_mesh(mesh, color="grey")

        # 添加淡淡的线框增强形状感
        plotter.add_mesh(mesh, style='wireframe', color='black', opacity=0.05)

        # 4. 姿态同步核心逻辑
        if current_camera_pos is not None:
            # 如果不是第一个窗口，则复用上一个窗口记录的相机姿态
            plotter.camera_position = current_camera_pos

        # 5. 显示窗口
        # return_cpos=True 极其重要：它会在窗口关闭时返回最后的相机参数
        # 这些参数是一个元组：(camera_location, focal_point, up_vector)
        current_camera_pos = plotter.show(return_cpos=True)

        # 关闭当前 plotter 释放资源
        plotter.close()


# --- 调用流程 ---
if __name__ == "__main__":
    # 数据路径
    data_dir = r"H:\paper\tooth_segentation\viewer_data\1FJ3MLSY_upper"
    path = Path(data_dir)
    obj_files = sorted(list(path.glob("*.obj")))

    loaded_meshes = []

    # 先批量读取，解耦加载与显示
    print("正在读取文件...")
    for obj_f in obj_files:
        v, f, c = load_custom_obj(str(obj_f))
        if v is not None:
            loaded_meshes.append({'name': obj_f.name, 'v': v, 'f': f, 'c': c})

    # 执行姿态同步可视化
    visualize_with_synced_pose(loaded_meshes)








