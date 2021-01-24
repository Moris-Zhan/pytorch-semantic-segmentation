import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# import torch.utils.model_zoo as model_zoo

# from torchvision import models
# import torchvision.models as models

import math
import numpy as np

class DeConvNet(nn.Module):
    
    def __init__(self, n_class):
        super(DeConvNet,self).__init__()

        self.input_size = 224
        self.name = 'DeconvNet'
        self.loss = torch.nn.CrossEntropyLoss() # NEVER FORGET TO CHANGE LOSS_NAME WHEN CHANGING THE LOSS
        self.loss_name = 'CrossEntropyLoss'
        self.n_class = n_class

        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16, kernel_size=4,stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight) #Xaviers Initialisation
        self.swish1= nn.ReLU()

        #Max Pool 1
        self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.swish2 = nn.ReLU()

        #Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.swish3 = nn.ReLU()

        ##### SWITCH TO DECONVOLUTION PART #####

        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.swish4=nn.ReLU()

        #Max UnPool 1
        self.maxunpool1=nn.MaxUnpool2d(kernel_size=2)

        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=5)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.swish5=nn.ReLU()

        #Max UnPool 2
        self.maxunpool2=nn.MaxUnpool2d(kernel_size=2)

        #DeConvolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=16,out_channels=self.n_class,kernel_size=4)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.swish6=nn.Sigmoid()

    def forward(self,x):
        out=self.conv1(x)
        out=self.swish1(out)
        size1 = out.size()
        out,indices1=self.maxpool1(out)
        out=self.conv2(out)
        out=self.swish2(out)
        size2 = out.size()
        out,indices2=self.maxpool2(out)
        out=self.conv3(out)
        out=self.swish3(out)

        out=self.deconv1(out)
        out=self.swish4(out)
        out=self.maxunpool1(out,indices2,size2)
        out=self.deconv2(out)
        out=self.swish5(out)
        out=self.maxunpool2(out,indices1,size1)
        out=self.deconv3(out)
        out=self.swish6(out)
        return(out)

    def criterion(self, output, label):
        return self.loss(output, label)


if __name__ == '__main__':     
    net = DeConvNet()
    print(net)