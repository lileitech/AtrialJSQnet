import numpy as np
import nibabel as nib
import torch
import os
import pylab as pl


Root_DIR = '/home/lilei/Workspace/AtrialJSQnet2021/Script_AJSQnet/lossfile'
lossfile1 = Root_DIR + '/laLoss_3d.txt'
lossfile2 = Root_DIR + '/scarLoss_3d.txt'
lossfile11 = Root_DIR + '/laLoss_3d_sdm.txt'
lossfile21 = Root_DIR + '/scarMaskLoss_1.txt'
lossfile22 = Root_DIR + '/scarMaskLoss_2.txt'

loss = np.loadtxt(lossfile1)
x = range(0, loss.size)
y = loss
pl.subplot(231)
pl.plot(x, y, 'g-', label='LA Seg loss')
pl.legend(frameon=False)


loss = np.loadtxt(lossfile11)
x = range(0, loss.size)
y = loss
pl.subplot(232)
pl.plot(x, y, 'g-', label='LA SDM loss')
pl.legend(frameon=False)


loss = np.loadtxt(lossfile2)
x = range(0, loss.size)
y = loss
pl.subplot(233)
pl.plot(x, y, 'g-', label='scar seg loss')
pl.legend(frameon=False)

loss = np.loadtxt(lossfile21)
x = range(0, loss.size)
y = loss
pl.subplot(234)
pl.plot(x, y, 'g-', label='mask loss 1')
pl.legend(frameon=False)


loss = np.loadtxt(lossfile22)
x = range(0, loss.size)
y = loss
pl.subplot(235)
pl.plot(x, y, 'g-', label='mask loss 2')
pl.legend(frameon=False)


pl.show()
pl.savefig("img.jpg") 