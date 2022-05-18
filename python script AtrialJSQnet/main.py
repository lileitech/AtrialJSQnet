import os
import time
import glob
import torch
import numpy as np
import nibabel as nib
import sys
from torch import optim
from torch.nn import DataParallel
from torch.backends import cudnn
import torch.utils.data as data
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from loaddata import LoadDataset, LoadDataset_scar, ProcessTestDataset
from network import Seg_3DNet, Seg_3DNet_2task
from function import F_loss, F_LoadParam_test, F_loss_SDM, F_loss_scar, F_LoadParam, F_mkdir

#Root_DIR = '/home/lilei/MICCAI2020/Data60/'
Root_DIR = '/home/lilei/Workspace/AtrialJSQnet2021/'
TRAIN_SAVE_DIR_best = Root_DIR + 'Script_AJSQnet/best_model/'

lossdir = Root_DIR + 'Script_AJSQnet/lossfile/'
lossfile1 = lossdir + 'laLoss_3d.txt'
lossfile2 = lossdir + 'scarLoss_3d.txt'
lossfile11 = lossdir + 'laLoss_3d_sdm.txt'
lossfile21 = lossdir + 'scarMaskLoss_1.txt'
lossfile22 = lossdir + 'scarMaskLoss_2.txt'

WORKERSNUM = 16
BatchSize = 2
NumEPOCH = 100
LEARNING_RATE = 1e-3
REGULAR_RATE = 0.96
WEIGHT_DECAY = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainingDataset(data.Dataset):
    def __init__(self, datapath):

        self.numpyimage = []
        self.numpylabel_LA = []
        self.numpylabel_LAdist = []
        self.numpyprob_normal = []
        self.numpyprob_scar = []
        self.numpyprob_background = []
        self.NumOfSubjects = 0

        self.datafile = glob.glob(datapath + '/*')
        for subjectid in range(len(self.datafile)):
            #if subjectid > 1:
            #     break     
            imagename = self.datafile[subjectid] + '/enhanced.nii.gz'
            LAlabelname = self.datafile[subjectid] + '/atriumSegImgMO.nii.gz'
            LAscarMaplabelname = self.datafile[subjectid] + '/scarSegImgM_wall.nii.gz'
            
            print('loading training image: ' + imagename) 
            
            numpyimage, numpylabel_LA, numpylabel_LAdist, numpyprob_normal, numpyprob_scar = LoadDataset_scar(imagename, LAlabelname, LAscarMaplabelname)
            self.numpyimage.extend(numpyimage)
            self.numpylabel_LA.extend(numpylabel_LA)
            self.numpylabel_LAdist.extend(numpylabel_LAdist)
            # self.numpyprob_background.extend(numpy2Dbackgroundprob)
            self.numpyprob_normal.extend(numpyprob_normal)
            self.numpyprob_scar.extend(numpyprob_scar)
            self.NumOfSubjects += 1

    def __getitem__(self, item):
        numpyimage = np.array([self.numpyimage[item]])
        numpylabel_LA = np.array([self.numpylabel_LA[item]])
        numpylabel_LA = (numpylabel_LA > 0) * 1
        numpylabel_LAdist = np.array([self.numpylabel_LAdist[item]])
        # numpyprob_background = np.array([self.numpyprob_background[item]])
        numpyprob_normal = np.array([self.numpyprob_normal[item]])
        numpyprob_scar = np.array([self.numpyprob_scar[item]])

        tensorimage = torch.from_numpy(numpyimage).float()
        tensorlabel_LA = torch.from_numpy(numpylabel_LA.astype(np.float32))
        tensorlabel_LAdist = torch.from_numpy(numpylabel_LAdist.astype(np.float32))
        # tensorprob_background = torch.from_numpy(numpyprob_background.astype(np.float32))
        tensorprob_normal = torch.from_numpy(numpyprob_normal.astype(np.float32))
        tensorprob_scar = torch.from_numpy(numpyprob_scar.astype(np.float32))

        return tensorimage, tensorlabel_LA, tensorlabel_LAdist, tensorprob_normal, tensorprob_scar

    def __len__(self):
        return self.NumOfSubjects

def Train_Validate(dataload, net, epoch, optimizer, savedir):
    start_time = time.time()
    flearning_rate = LEARNING_RATE*(REGULAR_RATE**(epoch//10))
    if flearning_rate<1e-5:
        flearning_rate = 1e-5
    f1 = open(lossfile1, 'a')
    f2 = open(lossfile2, 'a')
    f11 = open(lossfile11, 'a')
    f21 = open(lossfile21, 'a')
    f22 = open(lossfile22, 'a')

    net.train()

    for i, (lgeimage, lgelabel, lgedist, lgeprob_normal, lgeprob_scar) in enumerate(dataload):
        for param_group in optimizer.param_groups:
            param_group['lr'] = flearning_rate
        lgeimage, lgelabel, lgedist, lgeprob_normal, lgeprob_scar = lgeimage.to(device), lgelabel.to(device), lgedist.to(device), lgeprob_normal.to(device), lgeprob_scar.to(device)
        # lgeprob_background = lgeprob_background.to(device)
        optimizer.zero_grad()

        output = net(lgeimage)
        loss_la, loss_sdf_la, loss_scar, loss_scar_m1, loss_scar_m2 = F_loss_scar(output, lgelabel, lgedist, lgeprob_normal, lgeprob_scar)
        weight_sdm = 1e-2*(1.05**(epoch//10))
        weight_scar = 1e-2*(1.05**(epoch//10))
        loss = loss_la + weight_sdm*loss_sdf_la + 10*loss_scar # + 0.01*loss_scar_m1 + 0.01*loss_scar_m2

        loss.backward()
        optimizer.step()

        f1.write(str(loss_la.item()))
        f1.write('\n')

        f2.write(str(loss_scar.item()))
        f2.write('\n')

        f11.write(str(loss_sdf_la.item()))
        f11.write('\n')

        f21.write(str(loss_scar_m1.item()))
        f21.write('\n')

        f22.write(str(loss_scar_m2.item()))
        f22.write('\n')

        if i % 50 == 0:
            print('epoch %d , %d th, Seg-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, loss.item()))

    print('epoch %d , %d th, Seg-Net learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, loss.item()))
    strNetSaveName = 'net_with_%d.pkl' % epoch
    torch.save(net.state_dict(), os.path.join(savedir, strNetSaveName))
    end_time = time.time()
    print('---------------- Train Seg-Net: ' + strNetSaveName + ' , epoch %d cost time : %3.2f ----------------' % (epoch, end_time - start_time))

def main():
    is_for_training = True

    TRAIN_DIR_PATH = Root_DIR + 'Data60/train_data/'
    #TEST_DIR_PATH = Root_DIR + 'Data60/test_data_AtrialGeneral/ISBI2012/KCL/'
    #TEST_DIR_PATH = Root_DIR + 'Data60/test_data_AtrialGeneral/2018LAchallenge/Pre_ablation/'
    TEST_DIR_PATH = Root_DIR + 'Data60/test_data/'
    TRAIN_SAVE_DIR_Seg = Root_DIR + 'Script_AJSQnet/result_model/'

    if len(sys.argv) > 1:
        if sys.argv[1].find('train') != -1:
            is_for_training = True
        else:
            is_for_training = False

    if len(sys.argv) > 2:
        fold_name = sys.argv[2] #'12_3', '13_2', '23_1'
        TRAIN_DIR_PATH = TRAIN_DIR_PATH.replace('Data', 'Data_' + fold_name)
        TEST_DIR_PATH = TEST_DIR_PATH.replace('Data', 'Data_' + fold_name)
        TRAIN_SAVE_DIR_Seg = TRAIN_SAVE_DIR_Seg.replace('result_model', 'result_model_' + fold_name)
        F_mkdir(TRAIN_SAVE_DIR_Seg)


    if is_for_training:
        net = Seg_3DNet_2task(1, 1).to(device)
        dataset = TrainingDataset(TRAIN_DIR_PATH)
        data_loader = DataLoader(dataset, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True)     
        cudnn.benchmark = True
        #net = DataParallel(net,device_ids=[0,2,3])

        optimizer = optim.Adam(net.parameters())
        #optimizer = optim.SGD(net.parameters(), LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

        # Seg_net_param = TRAIN_SAVE_DIR_Seg + 'net_with_99.pkl'
        # # Seg_net_param = TRAIN_SAVE_DIR_best + 'net_with_99.pkl'
        # F_LoadParam(Seg_net_param, net)

        for epoch in range(NumEPOCH):
            #epoch = epoch + 100
            Train_Validate(data_loader, net, epoch, optimizer, TRAIN_SAVE_DIR_Seg)

    else:
        str_for_action = 'testing'
        print(str_for_action + ' .... ')
        net = Seg_3DNet_2task(1, 1).to(device)
        Seg_net_param = TRAIN_SAVE_DIR_Seg + 'net_with_99.pkl'
        F_LoadParam(Seg_net_param, net)
        net.eval()

        datafile = glob.glob(TEST_DIR_PATH + '/*')

        for subjectid in range(len(datafile)):
            imagename = datafile[subjectid] + '/enhanced.nii.gz'
            LAlabelname = datafile[subjectid] + '/atriumSegImgMO.nii.gz'
            LAscarMaplabelname =datafile[subjectid] + '/scarSegImgM_wall.nii.gz' 
            predict_LA, predict_scar = ProcessTestDataset(imagename, LAlabelname, LAscarMaplabelname, net)
            savedir = datafile[subjectid].replace('test_data', 'test_data_result')
            # savedir = datafile[subjectid].replace('Data_' + fold_name + '/test_data', 'fold_result')
            F_mkdir(savedir)
            nib.save(predict_LA, savedir + '/LA_predict_AJSQnet_SESA.nii.gz')
            nib.save(predict_scar, savedir + '/scar_predict_AJSQnet_SESA.nii.gz')

        print(str_for_action + ' end ')

if __name__ == '__main__':
    main()
