import torch 
import os
import torch.nn as nn
import torch.optim as optim
import math
import os
import torch.nn.functional as F
from unet_parts import *


""" Full assembly of the parts to form the complete network """


class UNet(nn.Module):
    def __init__(self, n_channels_rgb = 3, n_channels_depth = 1, n_classes = 2):
        super(UNet, self).__init__()
        self.n_channels_rgb = n_channels_rgb
        self.n_channels_depth = n_channels_depth
        self.n_classes = n_classes

        self.down1 = DownSample(in_channels=4, out_channels = 64)
        self.down2 = DownSample(in_channels= 64, out_channels= 128)
        self.down3 = DownSample(in_channels= 128, out_channels = 256)
        self.down4 = DownSample(in_channels= 256, out_channels= 512)
        self.up1 = UpSample(in_channels= 512, mid_channels= 1024, out_channels= 512, kernel_size= 3, Stride = 1)
        self.up2 = UpSample(in_channels= 1024, mid_channels = 512, out_channels= 256, kernel_size= 4, Stride = 2)
        self.up3 = UpSample(in_channels= 512, mid_channels= 256, out_channels= 128, kernel_size = 4, Stride= 2)
        self.up4 = UpSample(in_channels= 256, mid_channels= 128, out_channels = 64, kernel_size= 4, Stride= 2)
        self.up5 = UpSample(in_channels= 128 , mid_channels= 64 , out_channels= 32 , kernel_size=4 , Stride=2)
        self.outcome = OutSample(in_channels= 32, mid_channels= 256, out_channels = self.n_classes)



    def forward(self,rgb_batch, depth_batch):
        # rgb_batch , depth_batch = x,y
        combined_channel_input =  torch.cat((rgb_batch, depth_batch), dim= 1)   
        if torch.cuda.is_available(): combined_channel_input = combined_channel_input.to(gtx)
        print(combined_channel_input.size())
        #in: N,3,640,480  N,1,640,480        out: N,4,640,480
        x1 = self.down1(combined_channel_input)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.up1(x4)
        concatx4x5 = torch.cat((x4, x5), dim = 1)
        x6 = self.up2(concatx4x5)
        concatx3x6 = torch.cat((x3,x6),dim = 1)
        x7 = self.up3(concatx3x6)
        concatx2x7 = torch.cat((x2,x7), dim = 1)
        x8 = self.up4(concatx2x7)
        concatx1x8 = torch.cat((x1,x8), dim= 1)
        x9 = self.up5(concatx1x8) 
        logits = self.outcome(x9)
        

        if (False): #when not debugging set True to False
            print(f'x1: {x1.size()}')
            print(f'x2: {x2.size()}')
            print(f'x3: {x3.size()}')
            print(f'x4: {x4.size()}')
            print('x5: ',x5.size)
            print(f'x4x5: {concatx4x5.size()}')
            print(f'x6: {x6.size()}')
            print(f'x3x6: {concatx3x6.size()}')
            print(f'x7: {x7.size()}')
            print(f'x2x7: {concatx2x7.size()}')
            print(f'x8: {x8.size()}')
            print(f'x1x8: {concatx1x8.size()}')
            print(f'x9: {x9.size()}')
            print(f'logits: {logits.size()}')

        return logits


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits


if torch.cuda.is_available():
    print(f'GPU is available for usage')
    device = "cuda:0"
else: 
    print(f'No GPU detected')

gtx = torch.device(device)


network = UNet()
network.to(gtx)
sample_input_rgb = torch.rand(size = (5, 3, 640, 480))
# sample_input_rgb.to(gtx)
sample_input_depth = torch.rand(size = (5,1,640,480))
# sample_input_depth.to(gtx)
y_pred = network.forward(sample_input_rgb, sample_input_depth)
print(y_pred.size())