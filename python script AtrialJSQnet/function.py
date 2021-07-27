import numpy as np
from torch import nn
import torch
import collections
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import kornia
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def F_loss_scar(output, label, LAdist, prob_normal, prob_scar):
    out_LA, out_scar = output
    lossfunc1 = nn.BCELoss().to(device)
    loss_la = lossfunc1(out_LA, label)
    loss_sdf_la = torch.mean(((out_LA-0.5)*LAdist))

    lossfunc2 = nn.MSELoss().to(device)
    gt_scar_probmap = torch.cat((prob_normal, prob_scar), dim=1)
    loss_scar = lossfunc2(out_scar, gt_scar_probmap)#F_hellinger_distance

    lossfunc3 = nn.MSELoss(reduction='sum').to(device)
    mask_gd = ((prob_normal > 0.45) * (prob_normal < 0.5)).float() + ((prob_scar > 0.45) * (prob_scar < 0.5)).float()
    # mask_gd = (torch.min(torch.abs(torch.logit(gt_scar_probmap)), dim=1)[0]==0).float()
    # mask_gd = (torch.min(-torch.log(gt_scar_probmap), dim=1)[0]==0).float()
    # out_LA_gradient = kornia.sobel(((out_LA>0.5).float()))
    # mask_pred = (out_LA_gradient>0.4).float()
    mask_pred = ((out_LA > 0.1) * (out_LA < 0.8)).float()
    loss_scar_mask1 = lossfunc3(mask_gd*(gt_scar_probmap[:, 0] - gt_scar_probmap[:, 1]), mask_gd*(out_scar[:, 0] - out_scar[:, 1]))/torch.sum(mask_gd)
    loss_scar_mask2 = lossfunc3(mask_pred * (gt_scar_probmap[:, 0] - gt_scar_probmap[:, 1]), mask_pred * (out_scar[:, 0] - out_scar[:, 1])) / torch.sum(mask_pred)

    visualize_and_save = False
    if visualize_and_save == True:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.subplot(331)
        plt.imshow(label[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        plt.subplot(332)
        plt.imshow(LAdist[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray) #.cpu().detach().numpy()
        plt.subplot(333)
        plt.imshow(prob_normal[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        plt.subplot(334)
        plt.imshow((prob_scar)[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        plt.subplot(335)
        plt.imshow((out_LA)[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        plt.subplot(336)
        plt.imshow((out_scar)[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        plt.subplot(337)
        plt.imshow((out_scar)[0, 1, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        plt.subplot(338)
        plt.imshow((mask_gd)[0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        plt.subplot(339)
        plt.imshow((mask_pred)[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
        plt.show()
        plt.savefig('img.jpg')

    return loss_la, loss_sdf_la, loss_scar, loss_scar_mask1, loss_scar_mask2

def F_mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)

def F_hellinger_distance(p, q):
    """
    Calculates the hellinger's distance between two probability distributions.
    p --> probability vector 1.
    q --> probability vector 2. 
    """
    #d = torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q)) **2)) / np.sqrt(2)
    # d = torch.mean(((torch.sqrt(p) - torch.sqrt(q)) **2) / np.sqrt(2))
    lossfunc2 = nn.MSELoss().to(device)
    d = lossfunc2(torch.sqrt(p), torch.sqrt(q))/ np.sqrt(2)

    return d

def F_loss(output, label):

    lossfunc = nn.BCELoss().to(device)
    CE_loss = lossfunc(output, label)
    Dice = LabelDice(output, label, [0, 1])
    weightedDice = 10*torch.mean(1-Dice[:, 1]) + 0.1*torch.mean(1-Dice[:, 0])
    Dice_loss = 1-weightedDice
    loss = CE_loss + 0.1*Dice_loss

    return loss

def F_loss_SDM(output, label):
    lossfunc = nn.BCELoss().to(device)
    CE_loss = lossfunc(output, label)
    loss_seg = CE_loss

    gt_dis = compute_sdf(label.cpu().numpy(), output.shape)
    gt_dis = torch.from_numpy(gt_dis).float().to(device)
    loss_sdf_lei = torch.mean(((output - 0.5) * gt_dis))

    return loss_seg, loss_sdf_lei

def F_DistTransform(lab):
    posmask = lab.astype(np.bool)
    if posmask.any():
        negmask = ~posmask
        fg_dtm = distance(negmask)
    return fg_dtm

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
    T = 50
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = T*np.ones(out_shape) #np.zeros(out_shape)
    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                #sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf = negdis - posdis
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
                # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return np.clip(normalized_sdf, -T, T)

def AAAI_sdf_loss(net_output, gt_sdm):
    # print('net_output.shape, gt_sdm.shape', net_output.shape, gt_sdm.shape)
    # ([4, 1, 112, 112, 80])
    smooth = 1e-5
    # compute eq (4)
    intersect = torch.sum(net_output * gt_sdm)
    pd_sum = torch.sum(net_output ** 2)
    gt_sum = torch.sum(gt_sdm ** 2)
    L_product = (intersect + smooth) / (intersect + pd_sum + gt_sum + smooth)
    # print('L_product.shape', L_product.shape) (4,2)
    # L_SDF_AAAI = - L_product + torch.norm(net_output - gt_sdm, 1)/torch.numel(net_output)
    L_SDF_AAAI = torch.norm(net_output - gt_sdm, 1) / torch.numel(net_output)

    return L_SDF_AAAI

def LabelDice(A, B, class_labels):
    '''
    :param A: (n_batch, 1, n_1, ..., n_k)
    :param B: (n_batch, 1, n_1, ..., n_k)
    :param class_labels: list[n_class]
    :return: (n_batch, n_class)
    '''
    return F_Dice(torch.cat([1 - torch.clamp(torch.abs(A - i), 0, 1) for i in class_labels], 1),
                torch.cat([1 - torch.clamp(torch.abs(B - i), 0, 1) for i in class_labels], 1))


def F_DistTransformMap(img_gt):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """
    posmask = img_gt.astype(np.bool)
    if posmask.any():
        fg_dtm = distance(posmask)
    return fg_dtm

def F_Dice(A, B):
    '''
    A: (n_batch, n_class, ...)
    B: (n_batch, n_class, ...)
    return: (n_batch, n_class)
    '''
    eps = 1e-8
#    assert torch.sum(A * (1 - A)).abs().item() < eps and torch.sum(B * (1 - B)).abs().item() < eps
    A = A.flatten(2).float(); B = B.flatten(2).float()
    ABsum = A.sum(-1) + B.sum(-1)
    return 2 * torch.sum(A * B, -1) / (ABsum + eps)


#-----------------load net param-----------------------------
def F_LoadsubParam(net_param, sub_net, target_net):
    print(net_param)
    state_dict = torch.load(net_param, map_location='cpu')
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    sub_net.load_state_dict(new_state_dict)

    # ---------------load the param of Seg_net into SSM_net---------------
    sourceDict = sub_net.state_dict()
    targetDict = target_net.state_dict()
    target_net.load_state_dict({k: sourceDict[k] if k in sourceDict else targetDict[k] for k in targetDict})

def F_LoadParam(net_param, target_net):
    print(net_param)
    state_dict = torch.load(net_param, map_location='cpu')
    target_net.load_state_dict(state_dict)

def F_LoadParam_test(net_param, target_net):
    print(net_param)
    state_dict = torch.load(net_param, map_location='cpu')

    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    target_net.load_state_dict(new_state_dict)
