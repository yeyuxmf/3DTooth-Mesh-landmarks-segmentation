import os
import copy
import torch
import numpy as np
from data.data_processing import get_files
from config import config as cfg

from model.tooth_seg_landmark_net import ToothLandmark
from util import decoder_land





def read_data(file_path):


    file_data = np.load(file_path, allow_pickle=True).item()


    tooth_mask = np.zeros((cfg.tooth_nums, 2), np.int32)
    data_points = np.zeros((cfg.tooth_nums, cfg.sam_points, 3), np.float64)

    # random remove tooth
    for key in file_data:

        landmarks_ = file_data[key]["key_point"]
        mesh_points = np.array(file_data[key]["points"])
        se_index = np.random.randint(0, mesh_points.shape[0], cfg.sam_points)
        select_points = mesh_points[se_index][..., :3]



        tooth_mask[int(key) - 1, 0] = 1
        if key< 6 or key >11:
            tooth_mask[int(key) - 1, 1] = 1

        data_points[int(key) - 1] = select_points
    teeth_landmarks = {'Mesial':[], 'Distal':[], 'InnerPoint':[], 'OuterPoint':[], 'FacialPoint':[], 'Cusp':[]}
    for i in range(len(landmarks_)):
        kclass_, coord =  landmarks_[i]
        teeth_landmarks[kclass_].append(coord)


    return data_points, teeth_landmarks, tooth_mask, file_data




def get_fixed_cd(prelandmarks, fl_mask, conf_thresh):

    mask = fl_mask.squeeze()
    indices = torch.nonzero(mask.squeeze() > 0).squeeze().detach().cpu().numpy()
    TN = indices.shape[0]

    plandmarks = prelandmarks.reshape(16, -1, 7)


    precls = plandmarks[..., 6].sigmoid()
    plandmarks = plandmarks[..., :3]

    pro = precls.detach().cpu().numpy()
    pcoord = plandmarks.detach().cpu().numpy()


    return pro, pcoord, indices

def get_match_cd(pre_matchLandmarks, match_mask, conf_thresh):

    mask = match_mask.squeeze()
    indices = torch.nonzero(mask.squeeze() > 0).squeeze().detach().cpu().numpy()
    TN = indices.shape[0]

    matchLandmarks = pre_matchLandmarks.reshape(16, -1, 7)


    precls = matchLandmarks[..., 6].sigmoid()
    pred_sigmas = matchLandmarks[..., 3:6].sigmoid()
    prelandmarks = matchLandmarks[..., :3]


    pro = precls.detach().cpu().numpy()
    pcoord = prelandmarks.detach().cpu().numpy()



    return pro, pcoord, indices


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


if __name__ == "__main__":

    # Try to load models
    model = ToothLandmark()
    model_path = "./save_model/seg_land_model_best.pth"
    model_initial(model, model_path)
    model.cuda().float().eval()

    all_files =[]
    file_path = "F:/teethMICCAI2022/3DTeethLand_landmarks_train/"
    get_files(file_path, all_files, "__kpt.json")

    file_path ="E:/DataSet/mesh_tooth_landmark_data/seg_landmarks_train_data/train/"
    file_list = []
    get_files(file_path, file_list, ".npy")


    save_root = "E:/DataSet/mesh_tooth_landmark_data/seg_landmarks_train_data/tmp/"
    all_files = [os.path.basename(line).replace("__kpt.json", "") for line in all_files]
    for i in range(len(file_list)):

        file_path = file_list[i]
        data_name = os.path.basename(file_path).replace("_cv1.npy", "")
        if data_name in all_files:
            continue

        data_points, teeth_landmarks, tooth_point_mask, file_data = read_data(file_path)
        av = np.sum(tooth_point_mask)

        tooth_mask = tooth_point_mask[:, 0]

        # Decentering data
        cpoint = np.mean(data_points[tooth_mask >= 1].reshape(-1, 3), axis=0, keepdims=True)
        data_points[tooth_mask >= 1] = data_points[tooth_mask >= 1] - cpoint.reshape(1, 1, 3)


        data_points = torch.tensor(data_points).unsqueeze(dim=0)
        tooth_point_mask = torch.tensor(tooth_point_mask).unsqueeze(dim=0)


        batch_data = data_points.cuda().float()
        fl_mask = tooth_point_mask[..., 0].cuda().long()
        match_mask = tooth_point_mask[..., 1].cuda().long()
        nums = 0
        pred_all = {key: {} for key in (cfg.fxiedPorder + cfg.matchPorder)}
        pred_alld = {key: {} for key in (cfg.fxiedPorder + cfg.matchPorder)}
        with torch.no_grad():

            prelandmarks, matchLandmarks, heat_out, flandmarks, mLandmarks, seg_score= model(batch_data, fl_mask)
            prelandmarks, matchLandmarks = prelandmarks.squeeze(), matchLandmarks.squeeze()
            flandmarks, mLandmarks = flandmarks[-1].squeeze(), mLandmarks[-1].squeeze()

            fpro, fpcoord, findices = get_fixed_cd(prelandmarks, fl_mask, conf_thresh=0.1)
            ######################################################
            mpro, mpcoord, mindices = get_match_cd(matchLandmarks, match_mask, conf_thresh=0.1)

            #######################################################################

            fxiedPorder = ['Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint']
            for indx in range(16):
                score, land = fpro[indx], fpcoord[indx]

                land = land + cpoint
                if indx not in findices:
                    continue
                file_data[indx+1]["key_point"]=[]
                for coord, sc, cls in zip(land, score, fxiedPorder):

                    file_data[indx+1]["key_point"].append([cls, coord])

                if indx not in mindices:
                    continue
                mscore, mland = mpro[indx], mpcoord[indx]
                mask = mscore > 0.1
                mland = mland[mask]
                mland = mland + cpoint
                for coord, sc in zip(mland, mscore):

                    file_data[indx+1]["key_point"].append(["Cusp", coord])
            save_path = save_root + data_name + "_test.npy"
            np.save(save_path, file_data)
            print("over")


