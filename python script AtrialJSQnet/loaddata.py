import numpy as np
import nibabel as nib
import torch
from scipy import stats
from torch import nn
import scipy.ndimage as ndimage
from function import compute_sdf
import kornia
from function import F_DistTransform
from scipy.special import expit, logit
from skimage.exposure import match_histograms

np.seterr(divide='ignore', invalid='ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
height = depth = 192 # 192,208
length = 80 #64, 80
patch_size = (height, depth, length)

class ImageCenterCrop(object):
    def __init__(self, output_size=patch_size):
        self.output_size = output_size

    def __call__(self, image, label, label2):

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label2 = np.pad(label2, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label2 = label2[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return image, label, label2

class LabelCenterCrop(object):
    def __init__(self, output_size=patch_size):
        self.output_size = output_size

    def __call__(self, image, label, label2):

        center_label = label[:, :, int(label.shape[2]/2)]
        center_coord = np.floor(np.mean(np.stack(np.where(center_label > 0)), -1)).astype(np.int16)      
        center_x, center_y = center_coord

        image = F_nifity_imageCrop(image, center_coord) 
        label = F_nifity_imageCrop(label, center_coord)
        label2 = F_nifity_imageCrop(label2, center_coord)

        return image, label, label2

class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, image, label, label2):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        label2 = np.rot90(label2, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        label2 = np.flip(label2, axis=axis).copy()

        return image, label, label2

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size=patch_size):
        self.output_size = output_size

    def __call__(self, image, label, label2):

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label2 = np.pad(label2, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label_new = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image_new = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label_new2 = label2[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        
        return image_new, label_new, label_new2

def Loadimage(imagename):
 
    nibimage= nib.load(imagename)
    imagedata = nibimage.get_data()
    numpyimage = np.array(imagedata).squeeze()   

    return numpyimage

def DataPreprocessing3d(numpyimage, numpylabel, numpylabel2):
    
    numpyimage, numpylabel, numpylabel2 = RandomRotFlip()(numpyimage, numpylabel, numpylabel2)             
    numpyimage, numpylabel, numpylabel2 = LabelCenterCrop()(numpyimage, numpylabel, numpylabel2)  
    numpyimage = np.nan_to_num(stats.zscore(numpyimage))

    return numpyimage, numpylabel, numpylabel2

def F_nifity_imageCrop(numpyimage, center_coord):
    center_x, center_y = center_coord
    shape = numpyimage.shape
    numpyimagecrop = np.zeros((height, depth, shape[2]), dtype=np.float32)
    numpyimagecrop[0:height, 0:depth, :] = \
        numpyimage[int(center_x - height/ 2):int(center_x + height/ 2),
        int(center_y - depth/ 2):int(center_y + depth / 2), :]
    numpyimagecrop_new = numpyimagecrop
    # if numpyimage.shape[2] == length:
    #     numpyimagecrop_new = numpyimagecrop
    # elif numpyimage.shape[2] > length:
    #     numpyimagecrop_new = numpyimagecrop[:, :, (numpyimage.shape[2]-length):numpyimage.shape[2]]
    # else:
    #     pad_width = ((0, 0), (0, 0), (0, int((length - numpyimage.shape[2]))))
    #     numpyimagecrop_new = np.pad(numpyimagecrop, pad_width, 'constant')

    return numpyimagecrop_new

def LoadDataset_scar(imagenames, labelnames, scarlabelnames):
    niblabel = nib.load(labelnames)
    labeldata = niblabel.get_data()
    numpylabel = np.array(labeldata).squeeze()
    center_numpylabel = numpylabel[:, :, int(numpylabel.shape[2] / 2)]
    center_coord = np.floor(np.mean(np.stack(np.where(center_numpylabel > 0)), -1)).astype(np.int16)
    numpylabel_crop = F_nifity_imageCrop(numpylabel, center_coord)

    nibimage = nib.load(imagenames)
    imagedata = nibimage.get_data()
    numpyimage = np.array(imagedata).squeeze()
    numpyimage_crop = F_nifity_imageCrop(numpyimage, center_coord)  # crop image
    numpyimage_crop_processed = np.nan_to_num(stats.zscore(numpyimage_crop))
    
    nibscarlabel = nib.load(scarlabelnames)
    scarlabeldata = nibscarlabel.get_data()
    numpyscarlabel = np.array(scarlabeldata).squeeze()
    numpyscarlabel_crop = F_nifity_imageCrop(numpyscarlabel, center_coord)

    numpylabel_crop_new = np.expand_dims(numpylabel_crop, 0)
    numpylabel_crop_new = np.expand_dims(numpylabel_crop_new, 0)
    numpylabel_crop_new = (numpylabel_crop_new>0)*1
    gt_dis = compute_sdf(numpylabel_crop_new, numpylabel_crop_new.shape)
    gt_LA_dis = np.squeeze(gt_dis, axis=1)

    gt_dis_normal = F_DistTransform(numpyscarlabel_crop==421)
    gt_dis_scar = F_DistTransform(numpyscarlabel_crop==422)
    gt_dis_normal, gt_dis_scar = np.expand_dims(gt_dis_normal, 0), np.expand_dims(gt_dis_scar, 0)
    # m = 10
    # gt_dis_normal, gt_dis_scar = m**(-gt_dis_normal), m**(-gt_dis_scar)
    gt_dis_normal, gt_dis_scar = np.exp(-gt_dis_normal), np.exp(-gt_dis_scar)

    # gt_dis_background = F_DistTransform(numpylabel_new2==0)
    # gt_dis_background = np.expand_dims(gt_dis_background, 0)
    # gt_dis_background = np.exp(-gt_dis_background)
    # gt_dis_all = gt_dis_normal + gt_dis_scar + gt_dis_background
    # gt_dis_normal, gt_dis_scar, gt_dis_background = gt_dis_normal/gt_dis_all, gt_dis_scar/gt_dis_all, gt_dis_background/gt_dis_all

    # numpylabel_crop = np.expand_dims(np.expand_dims(numpylabel_new2, 0), 0)
    # gt_dis_normal = compute_sdf(numpylabel_crop==421, numpylabel_crop.shape)
    # gt_dis_scar = compute_sdf(numpylabel_crop==422, numpylabel_crop.shape)
    # gt_dis_normal, gt_dis_scar = np.squeeze(np.squeeze(gt_dis_normal, axis=1), axis=0), np.squeeze(np.squeeze(gt_dis_scar, axis=1), axis=0)
    # gt_dis_normal, gt_dis_scar = expit(-gt_dis_normal), expit(-gt_dis_scar)

    # gt_dis_background = compute_sdf(numpylabel_crop==0, numpylabel_crop.shape)
    # gt_dis_background = np.squeeze(np.squeeze(gt_dis_background, axis=1), axis=0)
    # gt_dis_background = expit(-gt_dis_background)
    # gt_dis_all = gt_dis_normal + gt_dis_scar + gt_dis_background
    # gt_dis_normal, gt_dis_scar, gt_dis_background = gt_dis_normal/gt_dis_all, gt_dis_scar/gt_dis_all, gt_dis_background/gt_dis_all


    return np.expand_dims(numpyimage_crop_processed, 0), np.expand_dims(numpylabel_crop, 0), gt_LA_dis, gt_dis_normal, gt_dis_scar

def LoadDataset(imagenames,labelnames):
 
    niblabel = nib.load(labelnames)
    labeldata = niblabel.get_data()
    numpylabel = np.array(labeldata).squeeze()   
    center_numpylabel = numpylabel[:, :, int(numpylabel.shape[2]/2)]
    center_coord = np.floor(np.mean(np.stack(np.where(center_numpylabel > 0)), -1)).astype(np.int16)
    numpylabel_crop = F_nifity_imageCrop(numpylabel, center_coord)
    
    nibimage = nib.load(imagenames)
    imagedata = nibimage.get_data()
    numpyimage = np.array(imagedata).squeeze()
    numpyimage_crop = F_nifity_imageCrop(numpyimage, center_coord)  # crop image
    numpyimage_crop_processed = np.nan_to_num(stats.zscore(numpyimage_crop))

    return np.expand_dims(numpyimage_crop_processed, 0), np.expand_dims(numpylabel_crop, 0)

def save_test_img(nibimage, outputlab):
    #extend the output to original size
    imagedata = nibimage.get_data()
    numpyimage_uncrop = np.array(imagedata).squeeze()
    size = numpyimage_uncrop.shape
    center_numpylabel = numpyimage_uncrop[:, :, int(size[2]/2)]
    center_coord = np.floor(np.mean(np.stack(np.where(center_numpylabel > 0)), -1)).astype(np.int16)
    center_x, center_y = center_coord

    pad_width = ((center_x - height//2, size[0] - center_x - height//2),
                (center_y - depth//2, size[1] - center_y - depth//2), (0, 0))
    outputlab_new = np.pad(outputlab, pad_width, 'constant')
    if size[2] > length:
        pad_width = ((center_x - height//2, size[0] - center_x - height//2),
                     (center_y - depth//2, size[1] - center_y - depth//2), (size[2]-length, 0))
    else:
        pad_width = ((center_x - height//2, size[0] - center_x - height//2),
                     (center_y - depth//2, size[1] - center_y - depth//2), (0, 0))

    if size[2] < length:
        outputlab_new = np.pad(outputlab[:, :, 0:size[2]], pad_width, 'constant')
    else:
        outputlab_new = np.pad(outputlab, pad_width, 'constant')
    predictlabel = nib.Nifti1Image(outputlab_new, nibimage.affine, nibimage.header)

    return predictlabel

def ProcessTestDataset(imagename, LAlabelname, LAscarMaplabelname, Seg_net):
    print('loading test image: ' + imagename)
    nibimage = nib.load(LAlabelname)
    numpyimage, _, _, _, _, NumSlice = LoadDataset_scar(imagename, LAlabelname, LAscarMaplabelname, DA = False, hist_matching = False)

    for sliceid in range(NumSlice):
        tensorimage = torch.from_numpy(np.array([numpyimage[sliceid]])).unsqueeze(0).float().to(device)
        output = Seg_net(tensorimage)

        out_LA, out_scar = output
        out_scar = out_scar*((out_scar>0.1).float()) #to filter non-wall pixels

        output1 = out_LA.squeeze().cpu().detach().numpy()
        outputlab = (output1 > 0.5) * 1
        outputlab = outputlab[:, :, np.newaxis]
        if sliceid == 0:
            label_LA = outputlab
        else:
            label_LA = np.concatenate((label_LA, outputlab), axis=-1)
        
        output2 = np.squeeze((out_scar).cpu().detach().numpy(), axis=0)
        output_new = np.argmax(output2, axis=0)
        outputlab = ((output_new == 0)*421 + (output_new == 1)*422)*(output2[1]>0) 
        outputlab = outputlab[:, :, np.newaxis]
        if sliceid == 0:
            label_scar = outputlab
        else:
            label_scar = np.concatenate((label_scar, outputlab), axis=-1)
        
        visualize_and_save = False
        if visualize_and_save == True:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.subplot(221)
            plt.imshow(tensorimage[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.gray)
            plt.subplot(222)
            plt.imshow((output_new == 0)[:, :], cmap=plt.cm.gray) #.cpu().detach().numpy()
            plt.subplot(223)
            plt.imshow((output_new == 1)[:, :], cmap=plt.cm.gray)
            plt.subplot(224)
            plt.imshow((output_new == 2)[:, :], cmap=plt.cm.gray)
            plt.savefig('img.jpg')


    # # #---------------------save the predicted label-------------------
    predict_LA = save_test_img(nibimage, label_LA)
    predict_scar = save_test_img(nibimage, label_scar)

    return predict_LA, predict_scar
