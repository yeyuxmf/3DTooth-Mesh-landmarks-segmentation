import os

import vtk
import json
import random
import trimesh
import pyglet
import numpy as np
from pathlib import Path
from trimesh.viewer import SceneViewer
kp_color_map = {
    "MCP": [0, 0, 139, 255], "DCP": [135, 206, 235, 255],
    "CEP": [255, 255, 0, 255], "FA": [255, 255, 255, 255],
    "IEP": [255, 182, 193, 255], "DeCP": [255, 165, 0, 255],
    "BCP": [200, 0, 0, 255], "LCP": [199, 21, 133, 255],
    "MBCP": [255, 100, 100, 255], "DBCP": [139, 0, 0, 255],
    "MLCP": [218, 112, 214, 255], "DLCP": [128, 0, 128, 255],
    "CFP": [50, 205, 50, 255], "WALA": [169, 169, 169, 255]
}

cls_teeth = {"1": 18, "2": 17, "3": 16, "4": 15, "5": 14, "6": 13, "7": 12, "8": 11,
             "9": 21, "10": 22, "11": 23, "12": 24, "13": 25, "14": 26, "15": 27, "16": 28,

             "32": 48, "31": 47, "30": 46, "29": 45, "28": 44, "27": 43, "26": 42, "25": 41,
             "24": 31, "23": 32, "22": 33, "21": 34, "20": 35, "19": 36, "18": 37, "17": 38}


tooth_lk = {18: ['MLCP', 'MBCP', 'DBCP'], 17: ['MLCP', 'MBCP', 'DBCP'] , 16: ['MLCP', 'MBCP', 'DBCP'],
            15: ['LCP', 'BCP', 'MCP', 'DCP'], 14: ['LCP', 'BCP', 'MCP', 'DCP'],
            13: ['MCP', 'DeCP', 'DCP'], 12: ['MCP', 'CEP', 'DCP'], 11: ['MCP', 'CEP', 'DCP'],
            21: ['MCP', 'CEP', 'DCP'], 22: ['MCP', 'CEP', 'DCP'], 23: ['MCP', 'DeCP', 'DCP'],
            24: ['LCP', 'BCP', 'MCP', 'DCP'], 25: ['LCP', 'BCP', 'MCP', 'DCP'],
            26: ['MLCP', 'MBCP', 'DBCP'], 27: ['MLCP', 'MBCP', 'DBCP'], 28: ['MLCP', 'MBCP', 'DBCP'],

            48: ['MLCP', 'MBCP', 'DBCP'], 47: ['MLCP', 'MBCP', 'DBCP'] , 46: ['MLCP', 'MBCP', 'DBCP'],
            45: ['LCP', 'BCP', 'MCP', 'DCP'], 44: ['LCP', 'BCP', 'MCP', 'DCP'],
            43: ['MCP', 'DeCP', 'DCP'], 42: ['MCP', 'CEP', 'DCP'], 41: ['MCP', 'CEP', 'DCP'],
            31: ['MCP', 'CEP', 'DCP'], 32: ['MCP', 'CEP', 'DCP'], 33: ['MCP', 'DeCP', 'DCP'],
            34: ['LCP', 'BCP', 'MCP', 'DCP'], 35: ['LCP', 'BCP', 'MCP', 'DCP'],
            36: ['MLCP', 'MBCP', 'DBCP'], 37: ['MLCP', 'MBCP', 'DBCP'], 38: ['MLCP', 'MBCP', 'DBCP'],
            }

tooth_id = {18: 1, 17: 2, 16: 3, 15: 4, 14: 5, 13: 6, 12: 7, 11: 8,
            21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
            48: 1, 47: 2, 46: 3, 45: 4, 44: 5, 43: 6, 42: 7, 41: 8,
            31: 9, 32: 10, 33: 11, 34: 12, 35: 13, 36: 14, 37: 15, 38: 16,
            }
class AlignmentConfig:
    dim: int = 3                # 空间维度
    teeth_nums: int = 32        # 总牙齿数
    sam_points: int = 300       # 每颗牙齿采样的点数

    uAntTeeth = [6,7,8,9,10,11]
    urposTeeth = [12,13,14,15,16]

# 使用方式
config = AlignmentConfig()
def FileErgodic(file_root, flie_list, type_):

    for file in os.listdir(file_root):
      newDir = os.path.join(file_root,file)
      if os.path.isdir(newDir):
          FileErgodic(newDir, flie_list, type_)
      else:
        if type_ in file:
            flie_list.append(newDir)

def read_stl(file_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_points(poly_data):
    points = poly_data.GetPoints()
    point_list = []
    for i in range(points.GetNumberOfPoints()):
        point = points.GetPoint(i)
        point_list.append(point)
    return np.array(point_list)


def read_data(data_path):

    lower_data = {}
    upper_data = {}
    dental_type = ["lower", "upper"]
    for ti in range (len(dental_type)):
        mesh_path = os.path.join(data_path, dental_type[ti])
        landmark_path = mesh_path +"/" + dental_type[ti] + "_landmarks.json"

        with open(landmark_path, 'r', encoding='utf-8') as f:
            feat_data = json.load(f)


        for tid in sorted(feat_data.keys(), key=int):
            path = os.path.join(mesh_path, f"{tid}.stl")
            if os.path.exists(path):
                mesh = trimesh.load_mesh(path)

                if dental_type[ti] == "upper":
                    upper_data[int(tid)] = [int(tid), feat_data[tid], mesh] #
                else:
                    lower_data[17-int(tid)] = [17-int(tid), feat_data[tid], mesh]  #

    # klp = ["MBCP","DBCP","MLCP","DLCP"]
    # cfp = [lower_data[2][1][kname] if kname in klp else np.array([0, 0, 0]) for ti, kname in enumerate(lower_data[2][1])]
    # cfp = np.sum(np.stack(cfp, axis=0), axis=0) /4.0
    # mesh_point = lower_data[2][2].vertices
    # diff = mesh_point - cfp.reshape(1, 3)
    # diff = np.sum(np.power(diff, 2), axis=-1)
    # index = np.argmin(diff)
    # cfp = mesh_point[index]
    # print(cfp)

    return lower_data, upper_data

def get_tooth_color(tooth_id):
    # 使用牙齿编号作为随机种子，保证每次运行颜色一致
    random.seed(int(tooth_id))
    return [random.randint(50, 255) for _ in range(3)] + [255]


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

def data_rot_trans(lower_data, upper_data, rot_matrix):

    utooth_points = []
    dtooth_points = []
    for i, tid in enumerate(upper_data):
        tid, tooth_kp, mesh = upper_data[tid]
        utooth_points.append(mesh.vertices)
    for i, tid in enumerate(lower_data):
        tid, tooth_kp, mesh = lower_data[tid]
        dtooth_points.append(mesh.vertices)

    ucp = np.mean(np.concatenate(utooth_points, axis=0), axis=0)
    dcp = np.mean(np.concatenate(dtooth_points, axis=0), axis=0)

    utranslation_matrix = trimesh.transformations.translation_matrix(-ucp)
    dtranslation_matrix = trimesh.transformations.translation_matrix(-dcp)

    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(180), [0, 0, 1])
    u_matrix = rotation_matrix @ rot_matrix  @ utranslation_matrix
    d_matrix = rot_matrix  @ dtranslation_matrix

    for i, tid in enumerate(upper_data):
        tid, key_point, mesh = upper_data[tid]
        mesh.apply_transform(u_matrix)

        for ki in key_point:
            key_point[ki] = np.dot(u_matrix[:3, :3], np.array(key_point[ki]))+ u_matrix[:3, 3]
        upper_data[tid] = [tid, key_point, mesh]


        #mesh.export("../outputs/" + str(tid) + "_upper.obj")
    for i, tid in enumerate(lower_data):
        tid, key_point, mesh = lower_data[tid]
        mesh.apply_transform(d_matrix)
        for ki in key_point:
            key_point[ki] = np.dot(d_matrix[:3, :3], np.array(key_point[ki]))+ d_matrix[:3, 3]
        lower_data[tid] = [tid, key_point, mesh]

        #mesh.export("../outputs/" + str(tid) + "_lower.obj")

    return lower_data, upper_data


def start_interactive_annotation(scene):
    class AnnotationViewer(SceneViewer):
        def __init__(self, scene):
            self.f_pressed = False
            self.picked_points = {}
            print("\n" + "=" * 50)
            print("交互式标注模式：")
            print("1. [标注] 按住 'F' + 鼠标左键点击模型")
            print("2. [查看] 红色小球将即时出现在点击处")
            print("=" * 50 + "\n")
            super().__init__(scene, start_loop=True)

        def on_key_press(self, symbol, modifiers):
            if symbol == 102:  # 'f' 键
                self.f_pressed = True

        def on_key_release(self, symbol, modifiers):
            if symbol == 102:
                self.f_pressed = False

        def get_ray_from_click(self, x, y):
            width, height = float(self.width), float(self.height)
            ndc_x = (2.0 * x) / width - 1.0
            ndc_y = (2.0 * y) / height - 1.0

            camera = self.scene.camera
            aspect = width / height
            fov_rad = np.radians(camera.fov[1])
            tan_half_fov = np.tan(fov_rad / 2.0)

            dir_cam = np.array([
                ndc_x * tan_half_fov * aspect,
                ndc_y * tan_half_fov,
                -1.0
            ])
            dir_cam /= np.linalg.norm(dir_cam)

            cam_pose = self.scene.camera_transform
            origin_world = cam_pose[:3, 3]
            direction_world = cam_pose[:3, :3] @ dir_cam

            return origin_world, direction_world

        def on_mouse_press(self, x, y, button, modifiers):
            if self.f_pressed and button == 1:
                origin_world, direction_world = self.get_ray_from_click(x, y)
                best_hit = {"dist": float('inf'), "point_local": None, "point_world": None, "node_name": None}

                for node_name in self.scene.graph.nodes_geometry:
                    if "marker" in node_name: continue

                    transform_world, geom_name = self.scene.graph[node_name]
                    mesh = self.scene.geometry[geom_name]
                    if not isinstance(mesh, trimesh.Trimesh): continue

                    inv_tf = np.linalg.inv(transform_world)
                    origin_local = (inv_tf @ np.append(origin_world, 1.0))[:3]
                    direction_local = inv_tf[:3, :3] @ direction_world

                    # 使用 intersects_location 更加稳健
                    locations, index_ray, index_tri = mesh.ray.intersects_location(
                        ray_origins=[origin_local],
                        ray_directions=[direction_local],
                        multiple_hits=False
                    )

                    if len(locations) > 0:
                        loc_local = locations[0]
                        loc_world = (transform_world @ np.append(loc_local, 1.0))[:3]
                        dist = np.linalg.norm(loc_world - origin_world)

                        if dist < best_hit["dist"]:
                            best_hit.update({
                                "dist": dist,
                                "point_local": loc_local,
                                "point_world": loc_world,
                                "node_name": node_name
                            })

                if best_hit["node_name"] is not None:
                    p_local = best_hit["point_local"]
                    p_world = best_hit["point_world"]

                    print(f"[OK] 击中: {best_hit['node_name']} -> Local: {p_local.round(3)}")

                    # 存储坐标
                    self.picked_points[f"p_{len(self.picked_points)}"] = p_local.tolist()

                    # --- 绘制红色小球 ---
                    # 根据模型尺寸调整半径，牙齿模型通常 0.2-0.5 比较合适
                    marker = trimesh.creation.uv_sphere(radius=0.3)
                    marker.visual.face_colors = [255, 0, 0, 255]

                    marker_name = f"marker_{len(self.scene.geometry)}"
                    self.scene.add_geometry(
                        marker,
                        node_name=marker_name,
                        transform=trimesh.transformations.translation_matrix(p_world)
                    )

                    # --- 强制刷新界面（修正 AttributeError） ---
                    # 这种组合通常能强制 Pyglet 重新渲染当前帧
                    if hasattr(self, 'on_draw'):
                        self.on_draw()
                else:
                    print("[MISS] 未击中表面")
                return

            # 如果没有按 F 键，执行默认的旋转平移
            super().on_mouse_press(x, y, button, modifiers)

    viewer = AnnotationViewer(scene)
    return viewer.picked_points




def main():
    #folder_path = "F:/teeth_arrangement_data/dental_tooth_data/20230402test/"#
    folder_path = "I:/mesh_tooth_data/oral_scan/"
    path = Path(folder_path)
    data_dir = [os.path.join(folder_path, d.name) for d in path.iterdir() if d.is_dir()]



    save_root = "I:/mesh_tooth_data/pred_landmarks_data/train_data/"


    for di in range(0, len(data_dir), 1):
        try:
            tooth_data = read_data(data_dir[di])
        except Exception:
            continue
        print(di, "    ", data_dir[di])

        lower_data, upper_data = tooth_data

        rot_matrix = data_ori_align(upper_data)

        tooth_data = data_rot_trans(lower_data, upper_data, rot_matrix)

        print("over")
        for type_, t_data in enumerate(tooth_data):

            jaw_points ={}
            arch_type = "lw" if 0==type_ else "up"
            save_path = save_root + arch_type + str(di) + ".npy"

            for i, tid in enumerate(t_data):
                tid, tooth_kp, mesh = t_data[tid]
                tooth_kp_ = {}
                for ti, kname in enumerate(tooth_kp):
                    if kname in kp_color_map:
                        tooth_kp_[kname] = tooth_kp[kname]

                if "WALA" in tooth_kp_:
                    del tooth_kp_["WALA"]
                jaw_points[tid] = [tooth_kp_, mesh.vertices]
            np.save(save_path, jaw_points)


            # visual_elements = []
            # for i, tid in enumerate(t_data):
            #     tid, tooth_kp, mesh = t_data[tid]
            #
            #     visual_elements.append(mesh)
            #     current_tooth_color = get_tooth_color(tid)
            #     # 遍历该牙齿的所有关键点
            #     # if "WALA" in tooth_kp:
            #     #     del tooth_kp["WALA"]
            #
            #     for kp_name, kp_pos in tooth_kp.items():
            #
            #         # 提取缩写，防止 kp_name 是 "MCP_1" 这种格式
            #         short_name = kp_name.split("_")[0].upper()
            #
            #         # 获取预定义颜色，如果没找到则默认使用白色
            #         point_color = kp_color_map.get(short_name, [255, 255, 255, 255])
            #
            #         sphere = trimesh.creation.uv_sphere(radius=0.5)
            #         sphere.visual.face_colors = point_color
            #
            #         # 平移并添加到 visual_elements
            #         translation = np.eye(4)
            #         translation[:3, 3] = kp_pos
            #         sphere.apply_transform(translation)
            #         visual_elements.append(sphere)
            #
            #         # 3. 打印调试信息（因为 trimesh 默认 viewer 很难直接在 3D 空间显示文字标签）
            #         #print(f"Tooth {tid} - Landmark: {kp_name} at {kp_pos}")
            #
            #     # 创建场景并显示
            # scene = trimesh.Scene(visual_elements)
            # #scene.show()  # 这会打开一个交互式窗口
            # start_interactive_annotation(scene)


if __name__ == "__main__":
    main()