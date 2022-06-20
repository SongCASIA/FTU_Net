import numpy as np
import torch
import torch.nn.functional as F

def mean_iou(pred, gt):
    smooth = 1e-5

    intersection = np.sum(np.abs(pred * gt))
    union = np.sum(np.abs(pred)) + np.sum(np.abs(gt)) - intersection

    IoU = (intersection + smooth) / (union + smooth)

    return IoU

def dice_coefficient(pred, gt):
    smooth = 1e-5

    intersection = np.sum(np.abs(pred * gt))
    union = np.sum(np.abs(pred)) + np.sum(np.abs(gt))

    dice = 2*(intersection) / (union + smooth)

    return dice

def rc_pre_F1_score(pred, gt):
    smooth = 1e-5
    TP = np.count_nonzero(pred & gt)
    TN = np.count_nonzero(~pred & ~gt)
    FP = np.count_nonzero(pred & ~gt)
    FN = np.count_nonzero(~pred & gt)

    recall = float(TP) / (float(TP + FN) + smooth)
    precision = float(TP) / (float(TP + FP) + smooth)
    F1_score = (2 * recall * precision) / (float(precision + recall) + smooth)

    return recall, precision, F1_score