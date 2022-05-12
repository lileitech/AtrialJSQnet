import numpy as np
from medpy.metric import hd95, assd, hd, dc, specificity, sensitivity, true_positive_rate, true_negative_rate #pip install medpy
from hausdorff import hausdorff_distance
from skimage import measure
'''
0/1 binary mask
'''

def calculate_binary_generalized_dice(y_true, y_pred, thres=421):
    dc_foreground = calculate_binary_dice(y_true, y_pred, thres)

    y_true_reversed = np.where(y_true == thres, 1, 0)
    y_pred_reversed = np.where(y_pred == thres, 1, 0)
    dc_background = calculate_binary_dice(y_true_reversed, y_pred_reversed)

    num_foreground = np.sum(np.where(y_true > thres, 1, 0))
    num_background = np.sum(np.where(y_true == thres, 1, 0))
    num_all = num_foreground + num_background

    return dc_foreground*num_foreground/num_all + dc_background*num_background/num_all

def calculate_binary_dice(y_true, y_pred, thres=0.5):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    y_true = np.where(y_true > thres, 1, 0)
    y_pred = np.where(y_pred > thres, 1, 0)
    return dc(y_pred, y_true)

def calculate_binary_assd(y_true, y_pred, thres=0.5, spacing=[1, 1, 1]):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    y_true = np.where(y_true > thres, 1, 0)
    y_pred = np.where(y_pred > thres, 1, 0)
    return assd(y_pred, y_true, spacing)

def calculate_binary_hd(y_true, y_pred,thres=0.5, spacing=[1, 1, 1]):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    y_true = np.where(y_true > thres, 1, 0)
    y_pred = np.where(y_pred > thres, 1, 0)
    return hd(y_pred, y_true, spacing)

def calculate_binary_spe(y_true, y_pred,thres=0.5):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    y_true = np.where(y_true > thres, 1, 0)
    y_pred = np.where(y_pred > thres, 1, 0)
    return specificity(y_pred, y_true)

def calculate_binary_sen(y_true, y_pred,thres=0.5):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    y_true = np.where(y_true > thres, 1, 0)
    y_pred = np.where(y_pred > thres, 1, 0)
    return sensitivity(y_pred, y_true)

def calculate_binary_acc(y_true, y_pred, thres=0.5):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    y_true = np.where(y_true > thres, 1, 0)
    y_pred = np.where(y_pred > thres, 1, 0)
    se = sensitivity(y_pred, y_true)
    sp = specificity(y_pred, y_true)
    TN = true_negative_rate(y_pred, y_true)
    TP = true_positive_rate(y_pred, y_true)
    acc = (TN + TP)/(TN/sp + TP/se)
    return acc