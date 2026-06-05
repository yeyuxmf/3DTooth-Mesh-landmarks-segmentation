import torch
import numpy as np

def weighting_DSC(y_pred, y_true, class_weights, class_nums):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    mdsc = np.zeros((class_nums))
    n_classes = class_nums

    # convert probability to one-hot code    
    max_idx = y_pred.long()
    one_hot = torch.zeros((y_pred.shape[0], class_nums))
    one_hot[range(y_pred.shape[0]), max_idx] = 1
    gt = torch.zeros((y_pred.shape[0], class_nums))
    gt[range(y_pred.shape[0]), y_true.long()] = 1

    for c in range(0, n_classes):
        pred_flat = one_hot[:, c].reshape(-1)
        true_flat = gt[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        mdsc[c]= ((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))
        
    return mdsc


def weighting_ACC(y_pred, y_true, class_weights, class_nums):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    Acc = np.zeros((class_nums))
    n_classes = class_nums

    # convert probability to one-hot code
    max_idx = y_pred.long()
    one_hot = torch.zeros((y_pred.shape[0], class_nums))
    one_hot[range(y_pred.shape[0]), max_idx] = 1
    gt = torch.zeros((y_pred.shape[0], class_nums))
    gt[range(y_pred.shape[0]), y_true.long()] = 1

    for c in range(0, n_classes):
        preds = one_hot[:, c].reshape(-1)
        labels = gt[:, c].reshape(-1)


        preds_cls = (preds == 1).float()  # 预测为当前类别的区域
        labels_cls = (labels == 1).float()  # 真实为当前类别的区域

        # 计算正确预测的数量
        correct_predictions = (preds_cls * labels_cls).sum()  # 交集
        total_predictions = labels_cls.sum()  # 预测为当前类别的总数

        # 计算准确率

        accuracy = (correct_predictions+1) / (total_predictions+1)



        Acc[c] = accuracy

    return Acc

def weighting_IOU(y_pred, y_true, class_weights, class_nums):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    Iou = np.zeros((class_nums))
    n_classes = class_nums

    # convert probability to one-hot code
    max_idx = y_pred.long()
    one_hot = torch.zeros((y_pred.shape[0], class_nums))
    one_hot[range(y_pred.shape[0]), max_idx] = 1
    gt = torch.zeros((y_pred.shape[0], class_nums))
    gt[range(y_pred.shape[0]), y_true.long()] = 1

    for c in range(0, n_classes):
        pred_flat = one_hot[:, c].reshape(-1)
        true_flat = gt[:, c].reshape(-1)

        intersection = (pred_flat * true_flat).sum()  # 交集
        union = pred_flat.sum() + true_flat.sum() - intersection  # 并集

        Iou[c] = (intersection+1)  / (union+1)

    return Iou

def weighting_SEN(y_pred, y_true, class_weights, class_nums):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    msen = np.zeros((class_nums))
    n_classes = class_nums

    # convert probability to one-hot code
    max_idx = y_pred.long()
    one_hot = torch.zeros((y_pred.shape[0], class_nums))
    one_hot[range(y_pred.shape[0]), max_idx] = 1
    gt = torch.zeros((y_pred.shape[0], class_nums))
    gt[range(y_pred.shape[0]), y_true.long()] = 1

    for c in range(0, n_classes):
        pred_flat = one_hot[:, c].reshape(-1)
        true_flat = gt[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        msen[c] = ((intersection + smooth) / (true_flat.sum() + smooth))
        
    return msen


def weighting_PPV(y_pred, y_true, class_weights, class_nums):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    mppv = 0.0

    n_classes = class_nums

    # convert probability to one-hot code
    max_idx = y_pred.long()
    one_hot = torch.zeros((y_pred.shape[0], class_nums))
    one_hot[range(y_pred.shape[0]), max_idx] = 1
    gt = torch.zeros((y_pred.shape[0], class_nums))
    gt[range(y_pred.shape[0]), y_true.long()] = 1

    for c in range(0, n_classes):
        pred_flat = one_hot[:, c].reshape(-1)
        true_flat = gt[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        mppv += w*((intersection + smooth) / (pred_flat.sum() + smooth))
        
    return mppv

   
def Generalized_Dice_Loss(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] probability
        y_true [n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    loss = 0.
    n_classes = y_pred.shape[-1]
    
    for c in range(0, n_classes):
        pred_flat = y_pred[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
       
        # with weight
        w = class_weights[c]/class_weights.sum()
        loss += w*(1 - ((2. * intersection + smooth) /
                         (pred_flat.sum() + true_flat.sum() + smooth)))
       
    return loss


def DSC(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    dsc = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
        
    return dsc


def SEN(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    sen = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))
            
        sen = np.asarray(sen)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            sen.append(((intersection + smooth) / (true_flat.sum() + smooth)))
            
        sen = np.asarray(sen)
        
    return sen


def PPV(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [npts, n_classes] one-hot code
        y_true [npts, n_classes] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[-1]
    ppv = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))
            
        ppv = np.asarray(ppv)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[:, c].reshape(-1)
            true_flat = y_true[:, c].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            ppv.append(((intersection + smooth) / (pred_flat.sum() + smooth)))
            
        ppv = np.asarray(ppv)
        
    return ppv



def accuracy_check2(y_pred, y_true):
    print('y_pred size: ', y_pred.size())
    #n_pts = y_pred.shape[1] * y_pred.shape[0] # multiply the number of points per mesh with batch size
    n_pts = y_pred.shape[1] 
    n_classes = y_pred.shape[-1]

    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=-1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(-1, max_idx, 1)

    acc = 0.0
    for c in range(0, n_classes):
        pred_flat = one_hot[:, :, c].reshape(-1)
        true_flat = y_true[:, :, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        #print('intersection: ', intersection)
        acc += intersection

    return acc/n_pts
