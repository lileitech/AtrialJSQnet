# --coding:utf-8--

# 3Dpatch 5*5*11, compute similarity fo two gray patches, which are generated and saved already.
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
# from scipy.ndimage.interpolation import rotate
import os
import SimpleITK as sitk
# import nibabel as nib
import numpy as np
import glob
import collections
import sys
import time

# TRAIN_DIR_PATH = '/home/zibin112/Li_Lei/exp_la_scar_MedAI18_v02/Dataset/Data_100_patch/train_data/'
# TEST_DIR_PATH = '/home/zibin112/Li_Lei/exp_la_scar_MedAI18_v02/Dataset/Data_100_patch/test_data/'
# TRAIN_SAVE_DIR = '/home/zibin112/Li_Lei/exp_la_scar_MedAI18_v02/gc_n_link/result_model_Data100/'

Root_DIR = '/home/lilei/MICCAI2020/'

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


BatchSize = 200  ###'每次训练到样本量，建议50-100比较合适'
WORKERSNUM = 120
LEARNING_RATE = 0.1 ####初始学习率， 建议0.1-0.01---0.01
REGULAR_RATE = 0.2 #0.6
WEIGHT_DECAY = 1e-4  #####正则化参数，建议1e-3~~1e-5之间
NumOfEPOCH = 10  ######训练循环次数， 10次左右足够，具体看是否收敛 ---15

class PatchFeatureNet(nn.Module):
    def __init__(self, patchsize):
        super(PatchFeatureNet, self).__init__()
        self.inputsize = 512 * (patchsize[0] // 8) * (patchsize[1] // 8) * (patchsize[2] // 8)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.L1 = nn.Linear(self.inputsize, 256)
        self.L2 = nn.Linear(256, 128)
        self.L3 = nn.Linear(128, 64)

        self.sigmoid = nn.Sigmoid()

    def forward(self, patch):
        xsize = patch.size()
        out = self.conv1(patch)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv5(out)
        out = self.relu(out)
        out = self.conv6(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv7(out)
        out = self.relu(out)
        out = out.view(xsize[0], -1)
        out = self.L1(out)
        out = self.relu(out)
        out = self.L2(out)
        out = self.relu(out)
        out = self.L3(out)

        out = self.sigmoid(out)

        return out

class SimilarityNet(nn.Module):
    def __init__(self, patchsize):
        super(SimilarityNet, self).__init__()
        self.PatchFeatureNet = PatchFeatureNet(patchsize)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self.L1 = nn.Linear(65, 64)
        self.L2 = nn.Linear(64, 32)
        self.L3 = nn.Linear(32, 1)

    def forward(self, patch1, patch2, dist):
        feature1 = self.PatchFeatureNet(patch1)
        feature2 = self.PatchFeatureNet(patch2)
        out = feature1 * feature2 + (1 - feature1) * (1 - feature2)
        dist = torch.exp(-dist)
        out = torch.cat((out, dist), dim=1)
        out = self.L1(out)
        out = self.relu(out)
        out = self.L2(out)
        out = self.relu(out)
        out = self.L3(out)

        out = self.sigmoid(out)

        return out

class PatchPair_Similarity(Dataset):
    def __init__(self, datapath, patchsize, phase='train'):
        # load data when initializing
        self.PatchSize = [13,13,17]
        self.PatchSize[0] = patchsize[0]
        self.PatchSize[1] = patchsize[1]
        self.PatchSize[2] = patchsize[2]
        StrPatchSize = str(self.PatchSize[0]) + '_' + str(self.PatchSize[1]) + '_' + str(self.PatchSize[2])
        self.datafile = glob.glob(datapath + '/*/Patch_' + StrPatchSize+ '_MS')
        self.numpypatch1 = []
        self.numpypatch2 = []
        self.distance = []
        self.similarity = []
        self.NumOfItems = 0
        for caseid in range(len(self.datafile)):
            #if caseid>0:
            #  break
            patchname1 = glob.glob(self.datafile[caseid] + '/Patch_N1.*')
            patchname2 = glob.glob(self.datafile[caseid] + '/Patch_N2.*')
            infoname = glob.glob(self.datafile[caseid] + '/Patch_N_info.txt')

            with open(infoname[0], 'r') as f:
                line = f.readline()
                NNode = int(line)
                print(infoname[0] + ' , nodes: ' + str(NNode))
                if NNode>30000:
                    continue

                itkpatch1 = sitk.ReadImage(patchname1[0])
                a = sitk.GetArrayFromImage(itkpatch1).astype(np.float32)
                if len(self.numpypatch1)==0:
                    self.numpypatch1 = np.array(a)
                else:
                    self.numpypatch1 = np.concatenate( (np.array(self.numpypatch1),np.array(a)), axis=1 )

                itkpatch2 = sitk.ReadImage(patchname2[0])
                a = sitk.GetArrayFromImage(itkpatch2).astype(np.float32)
                if len(self.numpypatch2)==0:
                    self.numpypatch2 = np.array(a)
                else:
                    self.numpypatch2 = np.concatenate( (np.array(self.numpypatch2),np.array(a)), axis=1 )

                self.NumOfItems = self.NumOfItems + NNode
                for i in range(NNode):
                    line = f.readline()
                    self.distance.append(float(line.split()[2]))
                    line = f.readline()
                    self.similarity.append(float(line.split()[2]))

        self.numpypatch1 = np.reshape( self.numpypatch1, (self.NumOfItems, self.PatchSize[0]*self.PatchSize[1]*self.PatchSize[2]))
        self.numpypatch2 = np.reshape( self.numpypatch2, (self.NumOfItems, self.PatchSize[0]*self.PatchSize[1]*self.PatchSize[2]))

    def __getitem__(self, item):
        numpypatch1 = np.array([self.numpypatch1[item]], np.float32)
        numpypatch1 = np.reshape(numpypatch1, (1, self.PatchSize[2], self.PatchSize[1], self.PatchSize[0]))

        numpypatch2 = np.array([self.numpypatch2[item]], np.float32)
        numpypatch2 = np.reshape(numpypatch2, (1, self.PatchSize[2], self.PatchSize[1], self.PatchSize[0]))

        distance = np.array([self.distance[item]], np.float32)
        similarity = np.array([self.similarity[item]], np.float32)

        return torch.from_numpy(numpypatch1), torch.from_numpy(numpypatch2), torch.from_numpy(distance), torch.from_numpy(similarity)

    def __len__(self):
        return self.NumOfItems

def Train_Validate(patchsize, dataloader, testdata, net, loss, epoch, optimizer, lr, savedir):
    start_time = time.time()
    vali_loss = 0
    flearning_rate = lr * (REGULAR_RATE ** epoch)
    fregular_rate = 1.0
    StrPatchSize = str(patchsize[0]) + '_' + str(patchsize[1]) + '_' + str(patchsize[2])
    for i, (patch1, patch2, dist, simi) in enumerate(dataloader):

        net.train()
        for param_group in optimizer.param_groups:
            param_group['lr'] = flearning_rate
        patch1, patch2 = patch1.to(device), patch2.to(device)       
        dist, simi = dist.to(device), simi.to(device)
        output = net(patch1, patch2, dist)
        loss_output = loss(output, simi)
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()
        if i % 500 == 0:
            if i > 1:
                flearning_rate = flearning_rate * fregular_rate
            print('epoch %d , %d th, learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, loss_output.item()))

    print('epoch %d , %d th, learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, loss_output.item()))
    strNetSaveName = 'net_with' + StrPatchSize + '_%03d_%03d.pkl' % (epoch, 1)
    torch.save(net.state_dict(), os.path.join(savedir, strNetSaveName))
    end_time = time.time()
    print('---------------- Train: ' + strNetSaveName + ' , epoch %d cost time : %3.2f ----------------' % (epoch, end_time - start_time))

def main():
    PatchSize = [13, 13, 17]  # default patch size
    StrPatchSize = '13_13_17'
    print('pyscript_nlink train/test sx sy sz\n')
    is_for_training = False

    if len(sys.argv) > 1:
        if sys.argv[1].find('train') != -1:
            is_for_training = True
        else:
            is_for_training = False
    str_for_training = ''

    if len(sys.argv) > 2:
        fold_name = sys.argv[2] #'12_3', '13_2', '23_1'
        TRAIN_DIR_PATH = Root_DIR + 'Data_60_patch/train_data/'
        TEST_DIR_PATH = Root_DIR + 'Data_60_patch/test_data/' 
        TRAIN_SAVE_DIR = Root_DIR + 'LearnGC/result_model_N/'

        TRAIN_DIR_PATH = TRAIN_DIR_PATH.replace('patch', 'patch_' + fold_name)
        TEST_DIR_PATH = TEST_DIR_PATH.replace('patch', 'patch_' + fold_name)
        TRAIN_SAVE_DIR_Seg = TRAIN_SAVE_DIR_Seg.replace('result_model_N', 'result_model_N_' + fold_name)
        F_mkdir(TRAIN_SAVE_DIR_Seg)

    # if len(sys.argv) > 4:
    #     PatchSize[0] = int(sys.argv[2])
    #     PatchSize[1] = int(sys.argv[3])
    #     PatchSize[2] = int(sys.argv[4])
    # StrPatchSize = str(PatchSize[0]) + '_' + str(PatchSize[1]) + '_' + str(PatchSize[2])  # StrPatchSize = '5_5_11'


    if is_for_training:
        str_for_training = 'training'
        print(str_for_training + ' .... ')
        net = SimilarityNet(PatchSize)
        net = net.to(device)
        cudnn.benchmark = True

        #net = DataParallel(net)

        optimizer = torch.optim.SGD(net.parameters(), LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
        loss = nn.MSELoss()
        loss = loss.to(device)
        traindataset = PatchPair_Similarity(TRAIN_DIR_PATH, PatchSize)
        train_loader = DataLoader( traindataset, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM, pin_memory=True)
        test_loader = 1  # DataLoader(testdataset, batch_size=BatchSize, shuffle=False, num_workers=WORKERSNUM, pin_memory=True)

        for epoch in range(NumOfEPOCH):
            Train_Validate(PatchSize, train_loader, test_loader, net, loss, epoch, optimizer, LEARNING_RATE, TRAIN_SAVE_DIR)

    else:
        str_for_training = 'testing'
        net_param = TRAIN_SAVE_DIR + 'net_with' + StrPatchSize + '_009_001.pkl'  # strNetSaveName = 'net_with' + StrPatchSize + '_%03d_%03d.pkl' % (epoch, 1)
        print(str_for_training + ' .... ')
        print(net_param)
        datafile = glob.glob(TEST_DIR_PATH + '/*/Patch_' + StrPatchSize+ '_MS')

        state_dict = torch.load(net_param)
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        cudnn.benchmark = True
        test_net = SimilarityNet(PatchSize)
        test_net.load_state_dict(new_state_dict)
        test_net = test_net.to(device)
        #test_net = DataParallel(test_net)
        test_net.eval()

        for fileid in range(len(datafile)):
            gd_weight=[]
            n_weights = []
            n_distance = []
            nodeids = []
            cellids = []
            NNode = 0
            niifiles = glob.glob(datafile[fileid] + '/Patch_N*.nii.gz')
            niiimage1 = sitk.ReadImage(niifiles[0])
            patchimage1 = sitk.GetArrayFromImage(niiimage1).astype(np.float32)
            shape1 = patchimage1.shape
            patchimage1 = patchimage1.reshape((-1,shape1[2]))
            niiimage2 = sitk.ReadImage(niifiles[1])
            patchimage2 = sitk.GetArrayFromImage(niiimage2).astype(np.float32)
            patchimage2 = patchimage2.reshape((-1,shape1[2]))

            weightfile = glob.glob(datafile[fileid] + '/Patch_N_info.txt')
            strPredictResultFile = weightfile[0].replace('Patch_N_info.txt','Patch' + StrPatchSize + '_N_info_Predict.txt')
            predict_error = 0.0
            i_error = 0

            T_batchsize = 100

            with open(weightfile[0], 'r') as fread:
                line = fread.readline()
                NNode = int(line)
                n_batch, last = divmod(NNode,T_batchsize)
                for i in range(NNode):
                    #id1, id2 = divmod(i, 30000)
                    linedistance = fread.readline()
                    lineweight = fread.readline()
                    distance = float(linedistance.split()[2])
                    gd_weight.append(float(lineweight.split()[2]))
                    n_distance.append(distance)
                    nodeids.append( int(lineweight.split()[0] ) )
                    cellids.append( int(lineweight.split()[1] ) )

                for i in range(n_batch+1):
                    t_batchsize = min((i+1)*T_batchsize,NNode)-i*T_batchsize
                    numpypatch1 = np.array(patchimage1[i*T_batchsize:min((i+1)*T_batchsize,NNode),:], np.float32)
                    numpypatch1 = np.reshape(numpypatch1, (t_batchsize, 1, PatchSize[2],PatchSize[1],PatchSize[0]) )
                    numpypatch1 = torch.from_numpy(numpypatch1)

                    numpypatch2 = np.array(patchimage2[i*T_batchsize:min((i+1)*T_batchsize,NNode),:], np.float32)
                    numpypatch2 = np.reshape(numpypatch2, (t_batchsize, 1, PatchSize[2],PatchSize[1],PatchSize[0]) )
                    numpypatch2 = torch.from_numpy(numpypatch2)


                    distance = np.array(n_distance[i*T_batchsize:min((i+1)*T_batchsize,NNode)], np.float32)
                    distance = np.expand_dims(distance, 1)
                    distance = torch.from_numpy(distance)

                    numpypatch1, numpypatch2, distance = numpypatch1.to(device), numpypatch2.to(device), distance.to(device) 
                    output = test_net(numpypatch1, numpypatch2, distance) 
                    out = output.data.cpu()
                    weight = out.numpy()

                    n_weights=n_weights+list(weight.reshape(-1))

                    predict_error += sum(abs(weight.reshape(-1) - np.array(gd_weight[i*T_batchsize:min((i+1)*T_batchsize,NNode)], np.float32)))
                        #i_error = i_error + 1

                print('predict error of this subject ' + datafile[fileid] + ' is: ' + str(predict_error /NNode))
                with open(strPredictResultFile, 'w') as fw:
                    fw.write(str(NNode) + '\n')
                    for i in range(NNode):
                        fw.write('\t' + str(nodeids[i]) + ' ' + str(cellids[i]) + ' ' + str(n_distance[i]) + '\n')
                        fw.write(str(nodeids[i]) + ' ' + str(cellids[i]) + ' ' + str(n_weights[i]) + '\n')


if __name__ == '__main__':
    main()
