
import os
import random
import torch
import trimesh
import numpy as np
from data.util import walkFileType, walkFile, data_ori_align, get_files, kp_color_map, get_tooth_color, sindata_rot_trans
from config import config as cfg




class AlignmentConfig:
    dim: int = 3                # 空间维度
    teeth_nums: int = 16        # 总牙齿数
    sam_points: int = 512       # 每颗牙齿采样的点数

    uAntTeeth = [6,7,8,9,10,11]
    urposTeeth = [12,13,14,15,16]

    # tooth_key = {16: ['MCP', 'DCP', 'MBCP', 'DBCP', 'MLCP', 'DLCP', 'FA', 'CFP'],
    #              15: ['MCP', 'DCP', 'MBCP', 'DBCP', 'MLCP', 'DLCP', 'FA', 'CFP'],
    #              14: ['MCP', 'DCP', 'MBCP', 'DBCP', 'MLCP', 'DLCP', 'FA', 'CFP'],
    #              13: ['MCP', 'DCP', 'BCP', 'LCP', 'FA', 'CFP'],
    #              12: ['MCP', 'DCP', 'BCP', 'LCP', 'FA', 'CFP'],
    #              11: ['MCP', 'DCP', 'DeCP', 'FA'],
    #              10: ['MCP', 'DCP', 'IEP', 'FA', 'CEP'],
    #              9: ['MCP', 'DCP', 'IEP', 'FA', 'CEP'],
    #              8: ['MCP', 'DCP', 'IEP', 'FA', 'CEP'],
    #              7: ['MCP', 'DCP', 'IEP', 'FA', 'CEP'],
    #              6: ['MCP', 'DCP', 'DeCP', 'FA'],
    #              5: ['MCP', 'DCP', 'BCP', 'LCP', 'FA', 'CFP'],
    #              4: ['MCP', 'DCP', 'BCP', 'LCP', 'FA', 'CFP'],
    #              3: ['MCP', 'DCP', 'MBCP', 'DBCP', 'MLCP', 'DLCP', 'FA', 'CFP'],
    #              2: ['MCP', 'DCP', 'MBCP', 'DBCP', 'MLCP', 'DLCP', 'FA', 'CFP'],
    #              1: ['MCP', 'DCP', 'MBCP', 'DBCP', 'MLCP', 'DLCP', 'FA', 'CFP']}

    tooth_key = {1: ['Mesial', 'Distal', 'Cusp', 'InnerPoint', 'OuterPoint', 'FacialPoint'],
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

    Ntooth_key = {1: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'MBCP', 'DBCP', 'MLCP', 'DLCP', 'CFP'],
                 2: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'MBCP', 'DBCP', 'MLCP', 'DLCP', 'CFP'],
                 3: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'MBCP', 'DBCP', 'MLCP', 'DLCP', 'CFP'],
                 4: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'BCP', 'LCP', 'CFP'],
                 5: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'BCP', 'LCP', 'CFP'],
                 6: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'DeCP'],
                 7: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'IEP'],
                 8: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'IEP'],
                 9: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'IEP'],
                 10: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'IEP'],
                 11: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'DeCP'],
                 12: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'BCP', 'LCP', 'CFP'],
                 13: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'BCP', 'LCP', 'CFP'],
                 14: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'MBCP', 'DBCP', 'MLCP', 'DLCP', 'CFP'],
                 15: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'MBCP', 'DBCP', 'MLCP', 'DLCP', 'CFP'],
                 16: ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint', 'MBCP', 'DBCP', 'MLCP', 'DLCP', 'CFP']
                 }

# 使用方式
config = AlignmentConfig()



def read_data_stl(file_list):

    lower_data = {}
    upper_data = {}


    # landmark_pathu = os.path.dirname(os.path.dirname(file_list[0])) + "/up/" + "up_landmarks_me.npy"
    # landmark_pathd = os.path.dirname(os.path.dirname(file_list[0])) + "/down/" + "down_landmarks_me.npy"
    # if os.path.exists(landmark_pathu):
    #     landmarku = np.load(landmark_pathu, allow_pickle=True).item()
    #     landmarkd = np.load(landmark_pathd, allow_pickle=True).item()
    #     landmark = landmarku | landmarkd

    for di in range(len(file_list)):
        path = os.path.join(file_list[di])
        teeth_nums = os.path.split(file_list[di])[-1].replace(".stl", "").split("_")[0]
        key_point = "feat_data[tid]"
        # if os.path.exists(landmark_pathu):
        #     key_point = landmark[int(teeth_nums)][1]

        if int(teeth_nums)>32:
            continue
        if os.path.exists(path):
            if int(teeth_nums) >= 17:
                offv = 16
                lower_data[int(teeth_nums) -offv] = [int(teeth_nums)-offv, key_point , trimesh.load_mesh(path)] #
            else:
                upper_data[int(teeth_nums)] = [int(teeth_nums), key_point, trimesh.load_mesh(path)]  #

    return lower_data, upper_data



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
        upper_data[tid] = [tid, key_point, mesh]


        #mesh.export("./outputs/" + str(tid) + "_upper.obj")
    for i, tid in enumerate(lower_data):
        tid, key_point, mesh = lower_data[tid]
        mesh.apply_transform(d_matrix)
        lower_data[tid] = [tid, key_point, mesh]

        #mesh.export("./outputs/" + str(tid) + "_lower.obj")

    return lower_data, upper_data, d_matrix, u_matrix

def trans_input_data(file_data):


    tooth_mask = np.zeros((cfg.tooth_nums), np.int32)
    data_points = np.zeros((cfg.tooth_nums, cfg.sam_points, 3), np.float32)

    teeth_nums = []
    for key in file_data:
        teeth_nums.append(key)
        tooth_mask[int(key) - 1] = 1
        mesh_points = np.array(file_data[key][2].vertices)
        se_index = np.random.randint(0, mesh_points.shape[0], cfg.sam_points)
        data_points[int(key) - 1] = mesh_points[se_index]

    return data_points, tooth_mask

def decentering_data(data_points, tooth_mask):

    cpoint = np.mean(data_points[tooth_mask >= 1].reshape(-1, 3), axis=0, keepdims=True)
    data_points[tooth_mask >= 1] = data_points[tooth_mask >= 1] - cpoint.reshape(1, 1, 3)

    return data_points, cpoint

def output_landmark(tooth_data, landmarks, mask, matrix, cpoint):

   index = np.where(mask>=1)[0]
   matrix_inv = np.linalg.inv(matrix)
   for ti in index:
       tid = ti + 1
       tid_, key_point_, mesh = tooth_data[tid]

       landmark = landmarks[ti].squeeze().detach().cpu().numpy().reshape(-1, 3) + cpoint
       landmark = np.concatenate([landmark, np.ones((landmark.shape[0], 1))], axis=-1)
       landmark = (matrix_inv.dot(landmark.T)).T[:, :3]
       mesh.apply_transform(matrix_inv)

       distances, indices = mesh.nearest.vertex(landmark)
       landmark = mesh.vertices[indices]

       key_ = config.tooth_key[tid]
       sorted_key_ = [k for k in cfg.fxiedPorder if k in key_]
       key_point = {sorted_key_[pi]: landmark[pi] for pi in range(landmark.shape[0])}
       tooth_data[tid] = [tid, key_point, mesh]

   return tooth_data

def data_view(t_data, flga=1):

    visual_elements = []
    for i, tid in enumerate(t_data):
        tid, tooth_kp, mesh = t_data[tid]

        visual_elements.append(mesh)
        # 遍历该牙齿的所有关键点
        # if "WALA" in tooth_kp:
        #     del tooth_kp["WALA"]

        for kp_name, kp_pos in tooth_kp.items():

            # 提取缩写，防止 kp_name 是 "MCP_1" 这种格式
            short_name = kp_name.split("_")[0]

            # 获取预定义颜色，如果没找到则默认使用白色
            point_color = kp_color_map.get(short_name, [255, 255, 255, 255])

            sphere = trimesh.creation.uv_sphere(radius=0.5)
            sphere.visual.face_colors = point_color

            # 平移并添加到 visual_elements
            translation = np.eye(4)
            translation[:3, 3] = kp_pos
            sphere.apply_transform(translation)
            visual_elements.append(sphere)

            # 3. 打印调试信息（因为 trimesh 默认 viewer 很难直接在 3D 空间显示文字标签）
            #print(f"Tooth {tid} - Landmark: {kp_name} at {kp_pos}")

        # 创建场景并显示
    scene = trimesh.Scene(visual_elements)
    scene.camera.perspective = False
    # 2. 设置正面视角
    # 根据你的截图，牙齿的正面通常对应 (np.pi/2, 0, 0) 或者 (0, 0, 0)
    # 我们使用 set_camera 自动计算距离，确保模型填满窗口
    if 1 ==flga:
        angles = (np.pi, 0, 0)
    else:
        angles = (0, 0, np.pi)
    scene.set_camera(angles=angles, distance=100, center=scene.centroid+10)

    scene.show()  # 这会打开一个交互式窗口


def test():


    data_path = "E:/DataSet/mesh_tooth_landmark_data/TeethAlign3D-dataset/"
    dir_list = []
    #walkFileType(data_path, dir_list, "_U")
    walkFile(data_path, dir_list)

    model = torch.jit.load('./save_model/tooth_landmarks.pt').cuda().eval().float()
    # from model.tooth_landmark_net import ToothLandmark
    # from main import model_initial
    # model = ToothLandmark()
    # model_path = "./outputs/final_model.pth"
    # model_initial(model, model_path)
    # model.cuda().eval().float()

    save_root =  ""
    for fi in range(0, len(dir_list)):
        file_list = []
        file_path = dir_list[fi]
        print(fi, "  ", file_path)

        # get_files(file_path, file_list, "_NoRoot.stl")
        # get_files(file_path.replace("_U", "_L"), file_list, "_NoRoot.stl")

        get_files(file_path +"/crown/", file_list, ".stl")

        tooth_data = read_data_stl(file_list)

        lower_data, upper_data = tooth_data


        # rot_matrix = data_ori_align(upper_data)
        # lower_data, upper_data, d_matrix, u_matrix = data_rot_trans(lower_data, upper_data, rot_matrix) #private data
        #
        u_matrix = data_ori_align(upper_data)
        d_matrix = data_ori_align(lower_data)
        upper_data, tooth_data_points, u_matrix = sindata_rot_trans(upper_data, u_matrix)
        lower_data, tooth_data_points, d_matrix = sindata_rot_trans(lower_data, d_matrix)

        d_points, d_mask = trans_input_data(lower_data)
        u_points, u_mask = trans_input_data(upper_data)


        #Decentering data
        d_points, d_cpoint = decentering_data(d_points, d_mask)
        u_points, u_cpoint = decentering_data(u_points, u_mask)

        with torch.no_grad():
            d_prelandmarks, _ = model(torch.tensor(d_points).unsqueeze(dim=0).cuda().float(), torch.tensor(d_mask).unsqueeze(dim=0).cuda().float())
            u_prelandmarks, _  = model(torch.tensor(u_points).unsqueeze(dim=0).cuda().float(), torch.tensor(u_mask).unsqueeze(dim=0).cuda().float())

        lower_data = output_landmark(lower_data, d_prelandmarks, d_mask, d_matrix, d_cpoint)
        upper_data = output_landmark(upper_data, u_prelandmarks, u_mask, u_matrix, u_cpoint)

        tooth_data = [lower_data, upper_data]

        data_view(upper_data, flga=1)
        data_view(lower_data, flga=0)


        lw_tooth_kps, up_tooth_kps = {}, {}
        for i, tid in enumerate(lower_data):
            tid, tooth_kp, mesh = lower_data[tid]
            lw_tooth_kps[tid +16] = [tid +16, tooth_kp]
        for i, tid in enumerate(upper_data):
            tid, tooth_kp, mesh = upper_data[tid]
            up_tooth_kps[tid] = [tid, tooth_kp]
        # np.save(file_path.replace("_U", "_L") + "/" + "lower_landmarks.npy", lw_tooth_kps)
        # np.save(file_path + "/" + "upper_landmarks.npy", up_tooth_kps)







if __name__ == "__main__":
    test()










