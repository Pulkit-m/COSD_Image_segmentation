
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F



class DownSample(nn.Module):
    """
    One block of downsampling process as in the jpeg file in the root dir
    [conv2d -->relu -->BatchNorm]--> [conv2d --> relu--> BatchNorm] --> Max Pool 2d
    """

    def __init__(self, in_channels, out_channels , kernel_size = 3):
        super(DownSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = out_channels
        self.kernel_size = kernel_size
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels= self.mid_channels
                               ,kernel_size= self.kernel_size, stride = 1, padding = (1,1)),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels = self.mid_channels, out_channels= self.out_channels
                                ,kernel_size = self.kernel_size, stride = 1, padding = (1,1)),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        

    def forward(self, x):
        return self.sequence(x)


class UpSample(nn.Module):
    """
        [conv2d-->relu-->batchnorm] --> [conv2d---> relu--> batchnorm] -->transpose_conv
    """

    def __init__(self, in_channels, out_channels, mid_channels = None , kernel_size =3, Stride = 2):
        super(UpSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.k = 3
        self.kernel = kernel_size
        self.stride = Stride

        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels= self.mid_channels, kernel_size= self.k, 
            padding= (1,1), stride= 1),
            nn.BatchNorm2d(num_features = self.mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels= self.mid_channels, out_channels= self.mid_channels, kernel_size = self.k,
            padding= (1,1), stride = 1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = self.mid_channels, out_channels= self.out_channels, kernel_size = self.kernel,
            padding = (1,1), stride = self.stride)
            # nn.BatchNorm2d(self.out_channels)
        )

    def forward(self,x):
        """
        where x is the concatenation product of previous block and skip connection
        """
        return self.sequence(x)


class OutSample(nn.Module):
    """ 3 convolutional layers and outputs logits. Apply softmax or any other activation function as per you usage to the output"""
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size =3):
        super(OutSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels # outchannels is probably the number of classes in the data
        self.mid_channels = mid_channels
        self.k = kernel_size

        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels= self.in_channels, out_channels= self.mid_channels, kernel_size= self.k
            , padding = (1,1), stride = 1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels= self.mid_channels, out_channels = self.mid_channels, kernel_size = self.k,
            padding= (1,1), stride = 1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels = self.mid_channels, out_channels= self.out_channels, kernel_size = 1
            ,padding = (0,0), stride = 1)
        )

    def forward(self, x):
        return self.sequence(x)

















#From gitHub
# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)


#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # if you have padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)