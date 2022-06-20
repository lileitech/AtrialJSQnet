import nibabel as nib
import numpy as np
import os
import glob
import math
import pandas as pd
import SimpleITK as sitk
import argparse
from metric import calculate_binary_dice, calculate_binary_assd, calculate_binary_hd, calculate_binary_sen, calculate_binary_spe, calculate_binary_acc, calculate_binary_generalized_dice

def load_img(image_name):
    niblabel = nib.load(image_name)
    labeldata = niblabel.get_fdata()
    numpylabel = np.array(labeldata).squeeze().astype(int)
    return numpylabel

def evaluate(args):

    foldtrain = args.datapath
    datafile = glob.glob(foldtrain + '/*')
    Casename_list = []
    LAcavity_Dicelist, LAcavity_ASDlist, LAcavity_HDlist = [], [], []
    LAscar_Acclist, LAscar_Spelist, LAscar_Senlist, LAscar_Dicelist, LAscar_GDicelist = [], [], [], [], []

    for subjectid in range(len(datafile)):
        CaseName = datafile[subjectid].replace(foldtrain, '').replace('\\', '')
        Casename_list.append(CaseName)
        script_path = os.getcwd()
        path = foldtrain + '/' + CaseName + '/'
        os.chdir(path)
        print('evaluating test data: ' + CaseName)

        if args.eval_LAcavity:
            gt_LAcavity_volume_name = 'atriumSegImgMO.nii.gz'
            pre_LAcavity_volume_name = args.pre_LAcavity_name
            # evaluate LA cavity segmentation: Dice, ASD, HD
            itkspacing = sitk.ReadImage(gt_LAcavity_volume_name).GetSpacing()
            gt_LAcavity_volume = load_img(gt_LAcavity_volume_name)
            pre_LAcavity_volume = load_img(pre_LAcavity_volume_name)
            LAcavity_Dice_3d = calculate_binary_dice(gt_LAcavity_volume, pre_LAcavity_volume)
            LAcavity_ASD_3d = calculate_binary_assd(gt_LAcavity_volume, pre_LAcavity_volume, spacing=itkspacing)
            LAcavity_HD_3d = calculate_binary_hd(gt_LAcavity_volume, pre_LAcavity_volume, spacing=itkspacing)
            LAcavity_Dicelist.append(LAcavity_Dice_3d)
            LAcavity_ASDlist.append(LAcavity_ASD_3d)
            LAcavity_HDlist.append(LAcavity_HD_3d)

        if args.eval_LAscar:
            pre_LAscar_volume_name = args.pre_LAscar_name
            target_image_name = 'enhanced.nii.gz'
            gt_LAcavity_volume_name = 'atriumSegImgMO.nii.gz'
            gt_LAscar_volume_name = 'scarSegImgM.nii.gz'
            gt_LAcavity_surface_name = 'LA_surface.nii.gz'
            gt_LAcavity_GaussianBlur_name = 'LA_label_GauiisanBlur_M.nii.gz'
            
            os.system(args.toolpath + 'zxhimageinfo ' + gt_LAcavity_volume_name + ' -i ' + gt_LAcavity_volume_name)
            os.system(args.toolpath + 'zxhimageinfo ' + gt_LAscar_volume_name + ' -i ' + gt_LAscar_volume_name)

            # generate target LA surface for scar projection
            # toolpath = script_path + 'tools/'
            a = os.path.exists(gt_LAcavity_GaussianBlur_name)
            if not os.path.exists(gt_LAcavity_GaussianBlur_name):
                os.system(args.toolpath + 'zxhimageop -int ' + gt_LAcavity_volume_name + ' -o ' + gt_LAcavity_GaussianBlur_name + ' -gau 4 -v 0 ')
            if not os.path.exists(gt_LAcavity_surface_name):
                # os.system(args.toolpath + 'zxhimageinfo ' + gt_LAcavity_volume_name + ' -i ' + gt_LAcavity_volume_name)
                os.system(args.toolpath + 'zxhboundary -i ' + gt_LAcavity_volume_name + ' -o ' + gt_LAcavity_surface_name + '  -R 1 420 -v 0 ')
            # generate projected  ground truth scar segmentation
            gt_LAscar_surface_name = gt_LAscar_volume_name.replace('.nii.gz', '_surface.nii.gz')
            if not os.path.exists(gt_LAscar_surface_name):
                os.system(args.toolpath + 'GenSurfaceScar ' + target_image_name + ' ' + gt_LAcavity_volume_name + ' ' + gt_LAscar_volume_name + ' ' + gt_LAscar_surface_name + ' ' + gt_LAcavity_surface_name + ' ' + gt_LAcavity_GaussianBlur_name)
            # generate projected predicted scar segmentation
            pre_LAscar_surface_name = pre_LAscar_volume_name.replace('.nii.gz', '_surface.nii.gz')
            os.system(args.toolpath + 'GenSurfaceScar ' + target_image_name + ' ' + gt_LAcavity_volume_name + ' ' + pre_LAscar_volume_name + ' ' + pre_LAscar_surface_name + ' ' + gt_LAcavity_surface_name + ' ' + gt_LAcavity_GaussianBlur_name)

            # evaluate LA scar quantification: Accuracy, ..., Dice, generalized Dice
            gt_LAscar_surface = load_img(gt_LAscar_surface_name)
            pre_LAscar_surface = load_img(pre_LAscar_surface_name)
            LAscar_Acc_3d = calculate_binary_acc(gt_LAscar_surface, pre_LAscar_surface, thres=421)
            LAscar_Spe_3d = calculate_binary_spe(gt_LAscar_surface, pre_LAscar_surface, thres=421)
            LAscar_Sen_3d = calculate_binary_sen(gt_LAscar_surface, pre_LAscar_surface, thres=421)
            LAscar_Dice_3d = calculate_binary_dice(gt_LAscar_surface, pre_LAscar_surface, thres=421)
            LAscar_GDice_3d = calculate_binary_generalized_dice(gt_LAscar_surface, pre_LAscar_surface, thres=421)
            LAscar_Acclist.append(LAscar_Acc_3d)
            LAscar_Spelist.append(LAscar_Spe_3d)
            LAscar_Senlist.append(LAscar_Sen_3d)
            LAscar_Dicelist.append(LAscar_Dice_3d)
            LAscar_GDicelist.append(LAscar_GDice_3d)

    if args.eval_LAcavity:
        list = {'Casename': Casename_list, 'LAcavity_Dice': LAcavity_Dicelist, 'LAcavity_ASD': LAcavity_ASDlist, 'LAcavity_HD': LAcavity_HDlist}
        df = pd.DataFrame(list)
        df.to_csv(script_path + '/LAcavity_evaluate_result.csv', encoding='gbk', index=False)

    if args.eval_LAscar:
        list = {'Casename': Casename_list, 'LAscar_Acc': LAscar_Acclist, 'LAscar_Spe': LAscar_Spelist, 'LAscar_Sen': LAscar_Senlist, 'LAscar_Dice': LAscar_Dicelist, 'LAscar_GDice': LAscar_GDicelist}
        df = pd.DataFrame(list)
        df.to_csv(script_path + '/LAscar_evaluate_result.csv', encoding='gbk', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./dataset/test_data')
    parser.add_argument('--toolpath', type=str, default='E:/tools/')
    parser.add_argument('--eval_LAcavity', type=bool, default=False)
    parser.add_argument('--eval_LAscar', type=bool, default=True)
    parser.add_argument('--pre_LAcavity_name', type=str, default='LA_predict.nii.gz')
    parser.add_argument('--pre_LAscar_name', type=str, default='scar_predict.nii.gz')
    args = parser.parse_args()

    evaluate(args)







 
    


