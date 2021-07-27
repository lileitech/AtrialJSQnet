import torch.nn as nn
import torch
import kornia
import torch.nn.functional as F

#----------------------Seg-Net----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Seg_Encoder(nn.Module):
    def __init__(self,in_ch):
        super(Seg_Encoder, self).__init__()

        self.conv1 = DoubleConv(in_ch, 16)
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.conv2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.conv3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool3d(2, stride=2)
        self.conv4 = DoubleConv(64, 128)
        self.pool4 = nn.MaxPool3d(2, stride=2)
        self.conv5 = DoubleConv(128, 256)

        # self.atten1 = SAlayer(16)
        # self.atten2 = SAlayer(32)
        # self.atten3 = SAlayer(64)
        # self.atten4 = SAlayer(128)

    def forward(self,x):
        c1=self.conv1(x)
        #c1=self.atten1(c1)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        #c2=self.atten2(c2)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        #c3=self.atten3(c3)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        #c4=self.atten4(c4)
        p4=self.pool4(c4)
        c5=self.conv5(p4)

        return c1, c2, c3, c4, c5

class Seg_Decoder_1(nn.Module):
    def __init__(self, out_ch):
        super(Seg_Decoder_1, self).__init__()
        self.up6 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv6 = DoubleConv(256, 128)
        self.up7 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv7 = DoubleConv(128, 64)
        self.up8 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv8 = DoubleConv(64, 32)
        self.up9 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.conv9 = DoubleConv(32, 16)
        self.conv10 = nn.Conv3d(16, out_ch, 1)

    def forward(self, input):
        c1, c2, c3, c4, c5 = input
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9, c1], dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        # out = nn.Sigmoid()(c10)
        # #out = nn.Softmax(dim=1)(c10)

        return c10

class Seg_Decoder_2(nn.Module):
    def __init__(self, out_ch):
        super(Seg_Decoder_2, self).__init__()
        self.up6 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv6 = DoubleConv(256, 128)
        self.up7 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv7 = DoubleConv(128, 64)
        self.up8 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv8 = DoubleConv(64, 32)
        self.up9 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.conv9 = DoubleConv(32, 16)
        self.conv10 = nn.Conv3d(16, out_ch, 1)

    def forward(self, input):
        c1, c2, c3, c4, c5 = input
        up_6 = self.up6(c5)
        c6=self.conv6(up_6)
        up_7=self.up7(c6)
        c7=self.conv7(up_7)
        up_8=self.up8(c7)
        c8=self.conv8(up_8)
        up_9=self.up9(c8)
        c9=self.conv9(up_9)
        c10=self.conv10(c9)

        return c10

class Seg_3DNet_2task(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Seg_3DNet_2task, self).__init__()
        self.E_feature = Seg_Encoder(in_ch)
        self.D_feature_1 = Seg_Decoder_1(out_ch)
        self.D_feature_2 = Seg_Decoder_1(out_ch+1)

    def forward(self,x):
        encoder_feature = self.E_feature(x)
        c10_LA = self.D_feature_1(encoder_feature)
        c10_scar = self.D_feature_2(encoder_feature)
        out_LA = nn.Sigmoid()(c10_LA)
        out_scar = nn.Sigmoid()(c10_scar)

        return out_LA, out_scar

class Seg_3DNet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Seg_3DNet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 16)
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.conv2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.conv3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool3d(2, stride=2)
        self.conv4 = DoubleConv(64, 128)
        self.pool4 = nn.MaxPool3d(2, stride=2)
        self.conv5 = DoubleConv(128, 256)
        self.up6 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv6 = DoubleConv(256, 128)
        self.up7 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv7 = DoubleConv(128, 64)
        self.up8 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.conv8 = DoubleConv(64, 32)
        self.up9 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.conv9 = DoubleConv(32, 16)
        self.conv10 = nn.Conv3d(16, out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9, c1], dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = nn.Sigmoid()(c10)
        #out = nn.Softmax(dim=1)(c10)

        return out



