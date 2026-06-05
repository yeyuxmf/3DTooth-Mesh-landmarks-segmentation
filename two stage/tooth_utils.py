
import os
import json
import math
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pointnet2_ops._ext as _ext
from evaluation import centroids_pred_to_gt_attribution, calculate_jaw_TLA, calculate_jaw_TSA, calculate_jaw_TIR
from sklearn.decomposition import PCA
import vtkmodules.all as vtk
vtk.vtkObject.GlobalWarningDisplayOff()
color_space = {
	  0:(119, 119, 119), 1:(50, 0, 0), 2:(255, 0, 0), 3:(150, 0, 0), 4:(50, 0, 200),
	  5:(50, 50, 0), 6:(50, 0, 50), 7:(150, 50, 0), 8:(50, 50, 200) ,
	  9:(50, 100, 0), 10:(50, 0, 100), 11:(150, 100, 0), 12:(50, 150, 200),
	  13:(50, 150, 0), 14:(50, 0, 150), 15:(150, 150, 0), 16:(50, 250, 200),
    17: (50, 0, 0), 18: (255, 0, 0), 19: (150, 0, 0), 20: (50, 0, 200),
    21: (50, 50, 0), 22: (50, 0, 50), 23: (150, 50, 0), 24: (50, 50, 200),
    25: (50, 100, 0), 26: (50, 0, 100), 27: (150, 100, 0), 28: (50, 150, 200),
    29: (50, 150, 0), 30: (50, 0, 150), 31: (150, 150, 0), 32: (50, 250, 200)
}
CONVERT_TABLE = {0 :"0",18: '1', 17: '2', 16: '3', 15: '4', 14: '5', 13: '6', 12: '7', 11: '8',
                 21: '9', 22: '10', 23: '11', 24: '12', 25: '13', 26: '14', 27: '15', 28: '16',
                 54: '505', 55: '504', 64: '512', 65: '513',
                 38: '17', 37: '18', 36: '19', 35: '20', 34: '21', 33: '22', 32: '23', 31: '24',
                 41: '25', 42: '26', 43: '27', 44: '28', 45: '29', 46: '30', 47: '31', 48: '32',
                 75: '520', 74: '521', 84: '528', 85: '529'}

CLAS_INDEX = {0 :"0",  8: '1', 7: '2', 6: '3', 5: '4', 4: '5', 3: '6', 2: '7', 1: '8',
                 9: '1', 10: '2', 11: '3', 12: '4', 13: '5', 14: '6', 15: '7', 16: '8',
                 24: '1', 23: '2', 22: '3', 21: '4', 20: '5', 19: '6', 18: '7', 17: '8',
                 25: '1', 26: '2', 27: '3', 28: '4', 29: '5', 30: '6', 31: '7', 32: '8'}

CLAS_RLINDEX = {8: '0', 7: '0', 6: '0', 5: '0', 4: '0', 3: '0', 2: '0', 1: '0',
                 9: '1', 10: '1', 11: '1', 12: '1', 13: '1', 14: '1', 15: '1', 16: '1',
                 24: '0', 23: '0', 22: '0', 21: '0', 20: '0', 19: '0', 18: '0', 17: '0',
                 25: '1', 26: '1', 27: '1', 28: '1', 29: '1', 30: '1', 31: '1', 32: '1'}


def FileErgodic(file_root, flie_list, type_):

    for file in os.listdir(file_root):
      newDir = os.path.join(file_root,file)
      if os.path.isdir(newDir):
          FileErgodic(newDir, flie_list, type_)
      else:
        if type_ in file:
            flie_list.append(newDir)

def read_teeth_mask(file_name):
    with open(file_name, "r") as file_:
        mask_json = json.load(file_)
        label = np.array(mask_json["labels"])
        instance = np.array(mask_json["instances"])

    nonv = np.unique(label[np.nonzero(label)])
    thids = {}
    for i in range(nonv.shape[0]):

        tindex = np.where(label == nonv[i])
        tid = CONVERT_TABLE[label[tindex[0]][0]]
        thids[(i, int(tid))] = tindex[0]
    label = np.array([int(CLAS_INDEX[int(CONVERT_TABLE[i])])  for i in label])

    return label, instance, thids

def orientation_determine(plyd, ori_verts, mean_nvec):

    pca = PCA(n_components=3, svd_solver='randomized')
    pca.fit(ori_verts)
    mat = pca.components_

    if np.dot(mat[2], mean_nvec) < 0:
        y_axis = - mat[2]
    else:
        y_axis = mat[2]

    z_axis = mat[1]
    x_axis = np.cross(y_axis, z_axis)

    y_axis = np.append(y_axis, 0.)
    z_axis = np.append(z_axis, 0.)
    x_axis = np.append(x_axis, 0.)

    trans_mat = np.stack([x_axis, y_axis, z_axis, np.array([0., 0., 0., 1.])], axis=0)

    mat = vtk.vtkMatrix4x4()
    for i in range(trans_mat.shape[0]):
        for j in range(trans_mat.shape[1]):
            mat.SetElement(i, j, trans_mat[i, j])

    transform = vtk.vtkTransform()
    transform.SetMatrix(mat)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(plyd)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    plyd = transformFilter.GetOutput()

    verts = []
    for i in range(plyd.GetNumberOfPoints()):
        verts.append(plyd.GetPoint(i))
    verts = np.array(verts)

    return plyd, verts
def read_data( pth):

    reader = vtk.vtkOBJReader()
    reader.SetFileName(pth)
    reader.Update()
    plyd = reader.GetOutput()

    row_points = []
    for i in range(plyd.GetNumberOfPoints()):
        row_points.append(plyd.GetPoint(i))
    row_points = np.array(row_points)



    x_min, x_max, y_min, y_max, z_min, z_max = plyd.GetBounds()
    cx = (x_max + x_min) / 2.
    cy = (y_max + y_min) / 2.
    cz = (z_max + z_min) / 2.

    transform = vtk.vtkTransform()
    transform.Translate(-cx, -cy, -cz)
    # transform.RotateX(15)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(plyd)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    plyd = transformFilter.GetOutput()





    ori_verts = []
    for i in range(plyd.GetNumberOfPoints()):
        ori_verts.append(plyd.GetPoint(i))
    ori_verts = np.array(ori_verts)

    nvec = vtk.vtkPolyDataNormals()
    nvec.SetInputData(plyd)
    nvec.SetSplitting(0)
    nvec.SetComputeCellNormals(0)
    nvec.Update()
    nvec_op = nvec.GetOutput().GetPointData().GetNormals()
    nvecs = []
    for i in range(nvec.GetOutput().GetNumberOfPoints()):
        nvecs.append([nvec_op.GetComponent(i, 0), nvec_op.GetComponent(i, 1), nvec_op.GetComponent(i, 2)])

    nvecs = np.array(nvecs)
    mean_nvec = np.mean(nvecs, axis=0)
    mean_nvec = mean_nvec / np.linalg.norm(mean_nvec)

    plyd, verts = orientation_determine(plyd, ori_verts, mean_nvec)


    transform = vtk.vtkTransform()

    transform.RotateX(15)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(plyd)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    plyd = transformFilter.GetOutput()

    faces = []
    for id in range(plyd.GetNumberOfCells()):
        p0_idx = plyd.GetCell(id).GetPointId(0)
        p1_idx = plyd.GetCell(id).GetPointId(1)
        p2_idx = plyd.GetCell(id).GetPointId(2)
        faces.append([p0_idx, p1_idx, p2_idx])
    faces = np.array(faces)

    ori_verts = []
    for i in range(plyd.GetNumberOfPoints()):
        ori_verts.append(plyd.GetPoint(i))
    ori_verts = np.array(ori_verts)


    return plyd, ori_verts, faces, row_points
def gen_img(camera, plyd, size):
    render = vtk.vtkRenderer()
    render.SetActiveCamera(camera)
    render.ResetCameraClippingRange()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(plyd)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    render.AddActor(actor)

    renWin = vtk.vtkRenderWindow()

    renWin.SetSize(size[0], size[1])
    renWin.AddRenderer(render)
    renWin.SetOffScreenRendering(1)
    renWin.Render()

    win2image = vtk.vtkWindowToImageFilter()
    win2image.ReadFrontBufferOff()
    win2image.SetInputBufferTypeToRGB()
    win2image.SetInput(renWin)
    win2image.Update()
    img_data = win2image.GetOutput()

    dimension = img_data.GetDimensions()
    img = []
    for y in range(dimension[1] - 1, -1, -1):
        row = []
        for x in range(dimension[0]):
            pixel = []
            for i in range(3):
                pixel.append(img_data.GetScalarComponentAsFloat(x, y, 0, i))
            row.append(pixel)
        img.append(row)
    img = np.array(img, dtype=np.uint8)


    depth_arr = vtk.vtkFloatArray()
    depth_arr.SetNumberOfComponents(size[0])
    depth_arr.SetNumberOfTuples(size[1])
    renWin.GetZbufferData(0, 0, size[0]-1, size[1]-1, depth_arr)
    render.SetRenderWindow(renWin)

    depth_size = depth_arr.GetSize()
    depth_img = np.zeros((depth_size), np.float32)
    for di in range(depth_size):
        depth_img[di] = 1- depth_arr.GetValue(di)
    depth_img = depth_img.reshape(size[0], size[1])[::-1, :]
    nonv = np.sort(np.unique(depth_img))
    depth_img = (depth_img - nonv[1])/(nonv[-1]- nonv[1])


    return img, depth_img


def get_rowpoints(pth):

    reader = vtk.vtkOBJReader()
    reader.SetFileName(pth)
    reader.Update()
    plyd = reader.GetOutput()

    row_points = []
    for i in range(plyd.GetNumberOfPoints()):
        row_points.append(plyd.GetPoint(i))
    row_points = np.array(row_points)

    return row_points


def get_data_to_mesh(file_path, img_size):
    polyData, verts, faces, row_points = read_data(file_path)

    x_min, x_max, y_min, y_max, z_min, z_max = polyData.GetBounds()

    camera = vtk.vtkCamera()
    camera.ParallelProjectionOn()
    # cxz = ((x_max + x_min) / 2., (y_max + y_min) / 2.)
    # camera.SetPosition(cxz[0], cxz[1], 50)
    # camera.SetFocalPoint(cxz[0], cxz[1], 0.)
    # camera.SetViewUp(0., 1., 0.)
    # x_range = x_max - x_min
    # y_range = y_max - y_min
    # longer_lenth = max(x_range, y_range)
    cxz = ((x_max + x_min) / 2., (z_max + z_min) / 2.)
    camera.SetPosition(cxz[0], 50., cxz[1])
    camera.SetFocalPoint(cxz[0], 0., cxz[1])
    camera.SetViewUp(0., 0., 1.)
    x_range = x_max - x_min
    z_range = z_max - z_min
    longer_lenth = max(x_range, z_range)

    camera.SetParallelScale(longer_lenth / 2.)

    n, f = camera.GetClippingRange()
    mat = camera.GetCompositeProjectionTransformMatrix(1, n, f)

    view_mat = []
    for i in range(4):
        row = []
        for j in range(4):
            row.append(mat.GetElement(i, j))
        view_mat.append(row)
    view_mat = np.array(view_mat)

    image, dept_img = gen_img(camera, polyData, img_size)

    _verts = verts.copy()
    _verts = np.concatenate([_verts, np.ones((_verts.shape[0], 1))], axis=-1)
    _verts = np.matmul(view_mat, _verts.T).T
    _verts[:, 1] = - _verts[:, 1]
    _verts[:, 0:2] = _verts[:, 0:2] * (img_size[0] - 1) / 2.0 + (img_size[0] - 1) / 2.0
    vert_coords = np.trunc(_verts[:, :3]).astype(float)

    dept_img[dept_img < 0] = 0


    return image, dept_img, vert_coords, verts, faces, row_points



def get_scale_faces(ori_verts, faces, t_index):
    ve = np.zeros((ori_verts.shape[0]), np.int32)
    ve[t_index] = 1
    faces1 = ve[faces[:, 0]]
    faces2 = ve[faces[:, 1]]
    faces3 = ve[faces[:, 2]]
    face_mask = faces1 & faces2 & faces3
    face_mask = face_mask.astype(np.bool_)

    # face_mask = face_index.astype(np.bool_)
    face_index = faces[face_mask]
    face_index = face_index.reshape(face_index.shape[0] * 3)

    mask = np.zeros((ori_verts.shape[0]), np.int32)
    mask[t_index] = np.arange(t_index.shape[0])

    face_index = mask[face_index]
    face_index = face_index.reshape((face_index.shape[0] // 3, 3))

    return face_index

def mapping_point_clound(pcoords, conv_coords, row_verts, faces, scalek=10):

    wh = pcoords[:,2:4] - pcoords[:,0:2]
    bbox = pcoords[:,:4].copy()
    bbox[:,:2] = bbox[:,:2] - scalek
    bbox[:,2:4] = bbox[:,2:4] + scalek

    teeth_points = []
    teeth_faces = []
    teeth_indexes = []
    vids = np.array([i for i in range(row_verts.shape[0])])
    for i in range(bbox.shape[0]):
        box = bbox[i]
        maskx1 = conv_coords[:, 0] > box[0]
        masky1 = conv_coords[:, 1] > box[1]
        maskx2 = conv_coords[:, 0] < box[2]
        masky2 = conv_coords[:, 1] < box[3]

        mask = maskx1 & masky1 & maskx2 & masky2

        con_points = conv_coords[mask]

        #zv
        # maskz = con_points[:, 2] < (np.min(con_points[:, 2]) + zv[i])

        tpoints = row_verts[mask]#[maskz]

        t_index = vids[mask]#[maskz]
        teeth_indexes.append(t_index)
        # face_index = get_scale_faces(row_verts, faces, t_index)
        teeth_points.append(tpoints)
        # teeth_faces.append(face_index)

    return teeth_points, teeth_indexes

def data_normallize(points):

    points_mean = np.mean(points, axis=0)
    points_ = points - points_mean

    points_max = np.max(np.abs(points_))

    points_ = points_ / points_max

    return points_

def  FurthestPointSampling(xyz, npoint):
    xyz = torch.tensor(xyz).unsqueeze(dim=0).cuda().float()
    index = _ext.furthest_point_sampling(xyz, npoint).squeeze().detach().cpu().numpy()

    return index

def findNearestNeighbors(input_points, labeled_points, pred):


    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(labeled_points)
    distances, indices = nbrs.kneighbors(input_points)

    pred = pred[indices]


    return np.mean(pred, axis=-1)




def avaluation_teeth2022(gt_label_dict, pred_label_dict):
    # gt_label_dict["mesh_vertices"] =  gt_label_dict["mesh_vertices"] / np.max(np.abs(gt_label_dict["mesh_vertices"]))

    gt_instances = gt_label_dict['instances']
    pred_instances = pred_label_dict['instances']

    pred_instance_label_dict ={}
    pthids = pred_label_dict['thids']
    for vi, index in enumerate(pthids):
        label = int(pthids[vi][0])
        tids = pthids[vi][1]

        gt_verts = gt_label_dict["mesh_vertices"][tids]
        gt_center = np.mean(gt_verts, axis=0)
        tooth_size  = np.sqrt(np.sum((gt_center - gt_verts) ** 2, axis=0))
        pred_instance_label_dict[str(vi)] = {"label": label, "centroid": gt_center, "tooth_size": tooth_size}


    gt_instance_label_dict = {}

    teeth_ids = gt_label_dict['thids']

    for vi, index in enumerate(teeth_ids):
        tid = int(CLAS_INDEX[index[1]])
        gt_verts = gt_label_dict["mesh_vertices"][teeth_ids[index]]
        gt_center = np.mean(gt_verts, axis=0)
        tooth_size  = np.sqrt(np.sum((gt_center - gt_verts) ** 2, axis=0))
        gt_instance_label_dict[str(vi)] = {"label": tid, "centroid": gt_center, "tooth_size": tooth_size}


    matching_dict = centroids_pred_to_gt_attribution(gt_instance_label_dict, pred_instance_label_dict)
    jaw_TLA = calculate_jaw_TLA(gt_instance_label_dict, pred_instance_label_dict, matching_dict)
    jaw_TIR = calculate_jaw_TIR(gt_instance_label_dict, pred_instance_label_dict, matching_dict)

    jaw_TSA = calculate_jaw_TSA(gt_instances, pred_instances)
    jaw_TLA = math.exp(-jaw_TLA)

    return  jaw_TLA, jaw_TIR, jaw_TSA