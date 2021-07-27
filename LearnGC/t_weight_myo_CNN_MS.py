
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.autograd import Variable
import os
import SimpleITK as sitk
import numpy as np
import glob
import sys
import collections
import time

Root_DIR = '/home/lilei/MICCAI2020/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BatchSize = 200#50
WORKERSNUM =120
LEARNING_RATE =0.01 #0.01
REGULAR_RATE = 0.6
WEIGHT_DECAY = 1e-4
NumEPOCH = 15

class PatchFeatureNet(nn.Module):
    def __init__(self,patchsize):
        super(PatchFeatureNet, self).__init__()
        self.inputsize = 512 * (patchsize[0] // 8) * (patchsize[1] // 8) * (patchsize[2] // 8)
        self.relu = nn.ReLU(inplace=True)
        self.conv1= nn.Conv3d(1,64,kernel_size=3,padding=1)
        self.conv2= nn.Conv3d(64,64,kernel_size=3,padding=1)
        self.conv3= nn.Conv3d(64,128,kernel_size=3,padding=1)
        self.conv4= nn.Conv3d(128,128,kernel_size=3,padding=1)
        self.conv5= nn.Conv3d(128,256,kernel_size=3,padding=1)
        self.conv6= nn.Conv3d(256,256,kernel_size=3,padding=1)
        self.conv7= nn.Conv3d(256,512,kernel_size=3,padding=1)
        self.maxpool=nn.MaxPool3d(kernel_size=2,stride=2)
        self.L1= nn.Linear(self.inputsize,256)
        self.L2= nn.Linear(256,128)
        self.L3= nn.Linear(128,64)

        self.sigmoid = nn.Sigmoid()


    def forward(self, patch):
        xsize= patch.size()
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
        out= out.view(xsize[0],-1)
        out= self.L1(out)
        out= self.relu(out)
        out = self.L2(out)
        out = self.relu(out)
        out = self.L3(out)

        out = self.sigmoid(out)

        return out

class MultiScaleNet(nn.Module):
    def __init__(self,patchsize):
        super(MultiScaleNet, self).__init__()
        self.PatchFeature1 = PatchFeatureNet(patchsize)
        self.PatchFeature2 = PatchFeatureNet(patchsize)
        self.PatchFeature3 = PatchFeatureNet(patchsize)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        self.L1 = nn.Linear(64*3, 64)#?????
        self.L2 = nn.Linear(64, 32)
        self.L3 = nn.Linear(32, 1)

    def forward(self, patch1, patch2, patch3):
        feature1 = self.PatchFeature1(patch1)
        feature2 = self.PatchFeature2(patch2)
        feature3 = self.PatchFeature3(patch3)
        out = torch.cat((feature1, feature2, feature3), dim=1) #?????
        out = self.L1(out)
        out = self.relu(out)
        out = self.L2(out)
        out = self.relu(out)
        out = self.L3(out)

        out = self.sigmoid(out)

        return out

class PatchScarProb(Dataset):
    def __init__(self, datapath, patchsize, phase='train'):

        self.PatchSize = [13, 13, 17]
        self.PatchSize[0] = patchsize[0]
        self.PatchSize[1] = patchsize[1]
        self.PatchSize[2] = patchsize[2]
        StrPatchSize = str(self.PatchSize[0]) + '_' + str(self.PatchSize[1]) + '_' + str(self.PatchSize[2])
        self.datafile = glob.glob(datapath + '/*/Patch_' + StrPatchSize+ '_MS')
        self.numpypatch1 = []
        self.numpypatch2 = []
        self.numpypatch3 = []
        self.prob = []
        self.NumOfItems = 0
        self.index = 1
        for subjectid in range(len(self.datafile)):
            #if subjectid > 5:
             #  break

            patchname1 = glob.glob(self.datafile[subjectid] + '/Patch_T_scale1.*')
            patchname2 = glob.glob(self.datafile[subjectid] + '/Patch_T_scale2.*')
            patchname3 = glob.glob(self.datafile[subjectid] + '/Patch_T_scale3.*')
            infoname = glob.glob(self.datafile[subjectid] + '/Patch_T_info_norm.txt')

            with open(infoname[0], 'r') as f:
                line = f.readline()
                NNode = int(line)
                print(infoname[0] + ' , nodes: ' + str(NNode))
                if NNode>30000:
                    continue
                for i in range(NNode):
                    line = f.readline()
                    self.prob.append(float(line.split()[1]))
                self.NumOfItems = self.NumOfItems + NNode

                itkpatch1 = sitk.ReadImage(patchname1[0])
                a = sitk.GetArrayFromImage(itkpatch1).astype(np.float32)
                if len(self.numpypatch1)==0:
                    self.numpypatch1 = np.array(a)
                else:
                    self.numpypatch1 = np.concatenate((np.array(self.numpypatch1),np.array(a)), axis=1)

                itkpatch2 = sitk.ReadImage(patchname2[0])
                a = sitk.GetArrayFromImage(itkpatch2).astype(np.float32)
                if len(self.numpypatch2)==0:
                    self.numpypatch2 = np.array(a)
                else:
                    self.numpypatch2 = np.concatenate((np.array(self.numpypatch2),np.array(a)), axis=1)

                itkpatch3 = sitk.ReadImage(patchname3[0])
                a = sitk.GetArrayFromImage(itkpatch3).astype(np.float32)
                if len(self.numpypatch3)==0:
                    self.numpypatch3 = np.array(a)
                else:
                    self.numpypatch3 = np.concatenate((np.array(self.numpypatch3),np.array(a)), axis=1)

        self.numpypatch1 = np.reshape(self.numpypatch1, (self.NumOfItems, self.PatchSize[0] * self.PatchSize[1] * self.PatchSize[2]))
        self.numpypatch2 = np.reshape(self.numpypatch2, (self.NumOfItems, self.PatchSize[0] * self.PatchSize[1] * self.PatchSize[2]))
        self.numpypatch3 = np.reshape(self.numpypatch3, (self.NumOfItems, self.PatchSize[0] * self.PatchSize[1] * self.PatchSize[2]))

    def __getitem__(self, item):
        nodeid = item
        numpypatch1 = np.array([self.numpypatch1[nodeid]], np.float32)
        numpypatch1 = np.reshape(numpypatch1, (1,self.PatchSize[2],self.PatchSize[1],self.PatchSize[0]))
        numpypatch2 = np.array([self.numpypatch2[nodeid]], np.float32)
        numpypatch2 = np.reshape(numpypatch2, (1,self.PatchSize[2],self.PatchSize[1],self.PatchSize[0]))
        numpypatch3 = np.array([self.numpypatch3[nodeid]], np.float32)
        numpypatch3 = np.reshape(numpypatch3, (1,self.PatchSize[2],self.PatchSize[1],self.PatchSize[0]))
        prob = np.array([self.prob[nodeid]], np.float32)

        return torch.from_numpy(numpypatch1),torch.from_numpy(numpypatch2),torch.from_numpy(numpypatch3),torch.from_numpy(prob)

    def __len__(self):
        return self.NumOfItems



def Train_Validate(patchsize,dataloader,testdata, net, loss, epoch, optimizer, lr, savedir):
    start_time = time.time()
    flearning_rate=lr*(REGULAR_RATE**epoch)
    fregular_rate = 1.0
    for i, (patch1,patch2,patch3,simi) in enumerate(dataloader):
        net.train()
        for param_group in optimizer.param_groups:
            param_group['lr'] = flearning_rate
        patch1, patch2, patch3 = patch1.to(device), patch2.to(device), patch3.to(device)
        simi = simi.to(device)
        output = net(patch1,patch2,patch3)
        loss_output = loss(output, simi)
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()
        if i % 1000 == 0:
            if i > 1:
                flearning_rate = flearning_rate * fregular_rate
            print('epoch %d , %d th, learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, loss_output.item()))

    print('epoch %d , %d th, learning_rate:%.10f , loss: %.10f' % (epoch, i, flearning_rate, loss_output.item()))
    StrPatchSize = str(patchsize[0]) + '_' + str(patchsize[1]) + '_' + str(patchsize[2])
    strNetSaveName = 'net_with'+StrPatchSize+'_%03d_%03d.pkl' % (epoch, 1)
    torch.save(net.state_dict(), os.path.join(savedir, strNetSaveName ))
    end_time = time.time()
    print('---------------- Train: ' + strNetSaveName + ' , epoch %d cost time : %3.2f ----------------' % (epoch, end_time - start_time))

def main():
    PatchSize = [13, 13, 17]
    StrPatchSize = '13_13_17'
    print('pyscript train/test sx sy sz\n')
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
        TRAIN_SAVE_DIR = Root_DIR + 'LearnGC/result_model_T_myo/'

        TRAIN_DIR_PATH = TRAIN_DIR_PATH.replace('patch', 'patch_' + fold_name)
        TEST_DIR_PATH = TEST_DIR_PATH.replace('patch', 'patch_' + fold_name)
        TRAIN_SAVE_DIR_Seg = TRAIN_SAVE_DIR_Seg.replace('result_model_T_myo', 'result_model_T_myo_' + fold_name)
        F_mkdir(TRAIN_SAVE_DIR_Seg)

    # if len(sys.argv) > 4:
    #     PatchSize[0] = int(sys.argv[2])
    #     PatchSize[1] = int(sys.argv[3])
    #     PatchSize[2] = int(sys.argv[4])
    # StrPatchSize = str(PatchSize[0]) + '_' + str(PatchSize[1]) + '_' + str(PatchSize[2])
    
    if is_for_training:
        str_for_training = 'training'
        print(str_for_training + ' .... ')
        net = MultiScaleNet(PatchSize)
        net = net.to(device)
        cudnn.benchmark = True

        #net = DataParallel(net)
        optimizer = torch.optim.SGD(net.parameters(), LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
        loss = nn.MSELoss()
        loss = loss.to(device)
        traindataset = PatchScarProb(TRAIN_DIR_PATH, PatchSize)
        train_loader = DataLoader(traindataset, batch_size=BatchSize, shuffle=True, num_workers=WORKERSNUM,pin_memory=True)
        test_loader = 1

        for epoch in range(NumEPOCH):
            Train_Validate(PatchSize, train_loader, test_loader, net, loss, epoch, optimizer, LEARNING_RATE,TRAIN_SAVE_DIR)
    else:
        str_for_training = 'testing'
        net_param = TRAIN_SAVE_DIR + 'net_with' + StrPatchSize + '_014_001.pkl'
        print(str_for_training + ' .... ')
        print(net_param)
        datafile = glob.glob(TEST_DIR_PATH + '/*/Patch_' + StrPatchSize+ '_MS')

        state_dict = torch.load(net_param)
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        cudnn.benchmark = True
        test_net = MultiScaleNet(PatchSize)
        test_net.load_state_dict(new_state_dict)
        test_net = test_net.to(device)
        #test_net = DataParallel(test_net)
        test_net.eval()

        for subjectid in range(len(datafile)):
            t_weights = []
            gd_weight = []
            nodeids = []
            NNode = 0
            patchname1 = glob.glob(datafile[subjectid] + '/Patch_T_scale1.*')
            patchname2 = glob.glob(datafile[subjectid] + '/Patch_T_scale2.*')
            patchname3 = glob.glob(datafile[subjectid] + '/Patch_T_scale3.*')
            weightfile = glob.glob(datafile[subjectid] + '/Patch_T_info_norm.txt')
            strPredictResultFile = weightfile[0].replace( 'Patch_T_info_norm.txt', 'Patch' + StrPatchSize + '_T_info_norm_Predict.txt')


            niiimage = sitk.ReadImage(patchname1[0])
            numpyimage = sitk.GetArrayFromImage(niiimage).astype(np.float32)
            shape1 = numpyimage.shape
            numpyimage1 = numpyimage.reshape((-1, shape1[2]))

            niiimage = sitk.ReadImage(patchname2[0])
            numpyimage = sitk.GetArrayFromImage(niiimage).astype(np.float32)
            shape1 = numpyimage.shape
            numpyimage2 = numpyimage.reshape((-1, shape1[2]))

            niiimage = sitk.ReadImage(patchname3[0])
            numpyimage = sitk.GetArrayFromImage(niiimage).astype(np.float32)
            shape1 = numpyimage.shape
            numpyimage3 = numpyimage.reshape((-1, shape1[2]))

            predict_error = 0.0
            i_error = 0
            T_batchsize = 100
            with open(weightfile[0], 'r') as fread:
                line = fread.readline()
                NNode = int(line)

                for i in range(NNode):
                    line = fread.readline()
                    nodeids.append(int(line.split()[0]))
                    gd_weight.append(float(line.split()[1]))

                for i in range(NNode // T_batchsize + 1):
                    t_batchsize = min((i + 1) * T_batchsize, NNode) - i * T_batchsize

                    numpypatch1 = np.array(numpyimage1[i * T_batchsize:min((i + 1) * T_batchsize, NNode), :], np.float32)
                    numpypatch1 = np.reshape(numpypatch1,(t_batchsize, 1, PatchSize[2], PatchSize[1], PatchSize[0]))
                    numpypatch1 = torch.from_numpy(numpypatch1)

                    numpypatch2 = np.array(numpyimage2[i * T_batchsize:min((i + 1) * T_batchsize, NNode), :], np.float32)
                    numpypatch2 = np.reshape(numpypatch2,(t_batchsize, 1, PatchSize[2], PatchSize[1], PatchSize[0]))
                    numpypatch2 = torch.from_numpy(numpypatch2)

                    numpypatch3 = np.array(numpyimage3[i * T_batchsize:min((i + 1) * T_batchsize, NNode), :], np.float32)
                    numpypatch3 = np.reshape(numpypatch3,(t_batchsize, 1, PatchSize[2], PatchSize[1], PatchSize[0]))
                    numpypatch3 = torch.from_numpy(numpypatch3)

                    numpypatch1,numpypatch2,numpypatch3 = numpypatch1.to(device),numpypatch2.to(device),numpypatch3.to(device)
                    output = test_net(numpypatch1,numpypatch2,numpypatch3)
                    out = output.data.cpu()
                    weight = out.numpy()
                    t_weights = t_weights + list(weight.reshape(-1))
                    predict_error += sum(abs(weight.reshape(-1) - np.array(gd_weight[i * T_batchsize:min((i + 1) * T_batchsize, NNode)],np.float32)))
            print('predict error of this subject ' + datafile[subjectid] + ' is: ' + str(predict_error / NNode))
            with open(strPredictResultFile, 'w') as fw:
                fw.write(str(NNode) + '\n')
                for i in range(NNode):
                    fw.write(str(nodeids[i]) + ' ' + str(t_weights[i]) + '\n')

    print(str_for_training + ' end ')


if __name__ == '__main__':
    main()
