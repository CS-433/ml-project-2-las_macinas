"""
Models: Unet and CNN models
"""
import torch
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self, channels = 32, kernel_size=3, batch_normalization = True, drop = True, prob = 0.3, dilation_mode = 1):
        super().__init__()

        if dilation_mode == 1:
            pad = "same"
            dilat = 1
        elif dilation_mode ==2:
            pad = 2
            dilat = 2
        else:
            print("wrong conv_mode")

        self.conv1 = nn.Conv2d(3, channels, kernel_size, padding=pad, dilation = dilat)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding="same")

        self.conv3 = nn.Conv2d(channels, 2*channels, kernel_size, stride=1, padding=pad, dilation = dilat)
        self.conv4 = nn.Conv2d(2*channels, 2*channels, kernel_size, stride=1, padding="same")

        self.conv5 = nn.Conv2d(2*channels, 4*channels, kernel_size, stride=1, padding=pad, dilation = dilat)
        self.conv6 = nn.Conv2d(4*channels, 4*channels, kernel_size, stride=1, padding="same")

        self.conv7 = nn.Conv2d(4*channels, 8*channels, kernel_size, stride=1, padding=pad, dilation = dilat)
        self.conv8 = nn.Conv2d(8*channels, 8*channels, kernel_size, stride=1, padding="same")
        # --------------------------------------------------- encoder end  --------------------------------------------------- #
        self.conv9 = nn.Conv2d(8*channels, 16*channels, kernel_size, stride=1, padding=pad, dilation = dilat)
        self.conv10 = nn.Conv2d(16*channels, 16*channels, kernel_size, stride=1, padding="same")
        # --------------------------------------------------- decoder start  --------------------------------------------------- #
        self.convT11 = nn.ConvTranspose2d(16*channels, 8*channels, kernel_size=(2,2), stride=2, padding=0, output_padding=0)
        self.conv12 = nn.Conv2d(16*channels, 8*channels, kernel_size, stride=1, padding=pad, dilation = dilat)
        self.conv13 = nn.Conv2d(8*channels, 8*channels, kernel_size, stride=1, padding="same")

        self.convT14 = nn.ConvTranspose2d(8*channels,  4*channels, kernel_size=(2,2), stride=2, padding=0, output_padding=0)
        self.conv15 = nn.Conv2d(8*channels, 4*channels, kernel_size, stride=1, padding=pad, dilation = dilat)
        self.conv16 = nn.Conv2d(4*channels, 4*channels, kernel_size, stride=1, padding="same")

        self.convT17 = nn.ConvTranspose2d(4*channels, 2*channels, kernel_size=(2,2), stride=2, padding=0, output_padding=0)
        self.conv18 = nn.Conv2d(4*channels, 2*channels, kernel_size, stride=1, padding=pad, dilation = dilat)
        self.conv19 = nn.Conv2d(2*channels, 2*channels, kernel_size, stride=1, padding="same")

        self.convT20 = nn.ConvTranspose2d(2*channels, channels, kernel_size=(2,2), stride=2, padding=0, output_padding=0)
        self.conv21 = nn.Conv2d(2*channels, channels, kernel_size, stride=1, padding=pad, dilation = dilat)
        self.conv22 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding="same")
        self.conv23 = nn.Conv2d(channels, 1, 1, stride=1, padding="same")
        
        self.bn1 = nn.InstanceNorm2d(1*channels)
        self.bn2 = nn.InstanceNorm2d(2*channels)
        self.bn3 = nn.InstanceNorm2d(4*channels)
        self.bn4 = nn.InstanceNorm2d(8*channels)
        self.bn5 = nn.InstanceNorm2d(16*channels)

        self.mp22 = nn.MaxPool2d(2,padding=0,dilation=1)

        self.batch_normalization = batch_normalization
        self.drop = drop
        self.dropout =  nn.Dropout(p=prob)
        self.act = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x, mode="val"):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        c2 = self.conv2(y)
        c2 = self.bn1(c2)
        c2 = self.act(c2)
        p2 = self.mp22(c2)

        y = self.conv3(p2)
        y = self.bn2(y)
        y = self.act(y)
        c4 = self.conv4(y)
        c4 = self.bn2(c4)
        c4 = self.act(c4)
        p4 = self.mp22(c4)
        if self.drop: p4 = self.dropout(p4)

        y = self.conv5(p4)
        y = self.bn3(y)
        y = self.act(y)
        c6 = self.conv6(y)
        c6 = self.bn3(c6)
        c6 = self.act(c6)
        p6 = self.mp22(c6)
        if self.drop: p6 = self.dropout(p6)

        y = self.conv7(p6)
        y = self.bn4(y)
        y = self.act(y)
        c8 = self.conv8(y)
        c8 = self.bn4(c8)
        c8 = self.act(c8)
        p8 = self.mp22(c8)
        if self.drop: p8 = self.dropout(p8)
        # --------------------------------------------------- encoder end  --------------------------------------------------- #
        y = self.conv9(p8)
        y = self.bn5(y)
        y = self.act(y)
        y = self.conv10(y)
        y = self.bn5(y)
        y = self.act(y)
        if self.drop: y = self.dropout(y)
        # --------------------------------------------------- decoder start  --------------------------------------------------- #
        y = self.act(self.convT11(y))
        crop11 = torch.cat([c8,y],dim=1)
        y = self.conv12(crop11)
        y = self.bn4(y)
        y = self.act(y)
        y = self.conv13(y)
        y = self.bn4(y)
        y = self.act(y)
        if self.drop: y = self.dropout(y)

        y = self.act(self.convT14(y))
        crop14 = torch.cat([c6,y],dim=1)
        y = self.conv15(crop14)
        y = self.bn3(y)
        y = self.act(y)
        y = self.conv16(y)
        y = self.bn3(y)
        y = self.act(y)
        if self.drop: y = self.dropout(y)

        y = self.act(self.convT17(y))
        crop17 = torch.cat([c4,y],dim=1)
        y = self.conv18(crop17)
        y = self.bn2(y)
        y = self.act(y)
        y = self.conv19(y)
        y = self.bn2(y)
        y = self.act(y)
        if self.drop: y = self.dropout(y)

        y = self.act(self.convT20(y))
        crop20 = torch.cat([c2,y],dim=1)
        y = self.conv21(crop20)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv22(y)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv23(y)
        
        y = self.out_act(y)
        y = torch.squeeze(y,dim = 1)
        return y


class CNN(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=64, stride=0, batch_normalization = True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding="same")
        
        self.fc1 = nn.Linear(in_features=1024, out_features=512, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=2, bias=True)

        self.act = nn.ReLU()
        self.out_act = nn.Softmax()

        self.batch_normalization = batch_normalization
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_normalization: x = self.bn32(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.batch_normalization: x = self.bn64(x)
        x = self.act(x)
        x = self.pool(x)
        
        x = torch.flatten(x,start_dim=1, end_dim= -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out_act(x)

        return x

