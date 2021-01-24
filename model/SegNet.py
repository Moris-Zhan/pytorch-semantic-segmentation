import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class SegNetxxx(nn.Module):
    """Segnet network."""

    def __init__(self, nb_input_channels, n_classes, mean, std, orientation,
                 resolution, matrix_size, bn_momentum=0.1, drop_rate=0.4):
        """Init fields."""
        super(SegNet, self).__init__()

        self.input_nbr = nb_input_channels
        self.mean = mean
        self.std = std
        self.orientation = orientation
        self.resolution = resolution
        self.matrix_size = matrix_size
        # self.class_names = class_names
        # label_nbr = 1
        # if n_classes>1:
        #     label_nbr=n_classes+1
        label_nbr=n_classes


        self.conv11 = nn.Conv2d(nb_input_channels, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.drop11 = nn.Dropout2d(drop_rate)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.drop12 = nn.Dropout2d(drop_rate)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.drop21 = nn.Dropout2d(drop_rate)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.drop22 = nn.Dropout2d(drop_rate)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.drop31 = nn.Dropout2d(drop_rate)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.drop32 = nn.Dropout2d(drop_rate)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.drop33 = nn.Dropout2d(drop_rate)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop41 = nn.Dropout2d(drop_rate)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop42 = nn.Dropout2d(drop_rate)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop43 = nn.Dropout2d(drop_rate)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop51 = nn.Dropout2d(drop_rate)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop52 = nn.Dropout2d(drop_rate)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop53 = nn.Dropout2d(drop_rate)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop53d = nn.Dropout2d(drop_rate)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop52d = nn.Dropout2d(drop_rate)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop51d = nn.Dropout2d(drop_rate)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop43d = nn.Dropout2d(drop_rate)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.drop42d = nn.Dropout2d(drop_rate)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.drop41d = nn.Dropout2d(drop_rate)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.drop33d = nn.Dropout2d(drop_rate)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.drop32d = nn.Dropout2d(drop_rate)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.drop31d = nn.Dropout2d(drop_rate)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.drop22d = nn.Dropout2d(drop_rate)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.drop21d = nn.Dropout2d(drop_rate)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.drop12d = nn.Dropout2d(drop_rate)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

    def forward(self, x):
        """Forward method."""
        # normalization
        x = (x-self.mean)/self.std

        # Stage 1
        x11 = F.relu(self.drop11(self.bn11(self.conv11(x))))
        x12 = F.relu(self.drop12(self.bn12(self.conv12(x11))))
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)
        size1 = x12.size()

        # Stage 2
        x21 = F.relu(self.drop21(self.bn21(self.conv21(x1p))))
        x22 = F.relu(self.drop22(self.bn22(self.conv22(x21))))
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)
        size2 = x22.size()
        # Stage 3
        x31 = F.relu(self.drop31(self.bn31(self.conv31(x2p))))
        x32 = F.relu(self.drop32(self.bn32(self.conv32(x31))))
        x33 = F.relu(self.drop33(self.bn33(self.conv33(x32))))
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)
        size3 = x33.size()

        # Stage 4
        x41 = F.relu(self.drop41(self.bn41(self.conv41(x3p))))
        x42 = F.relu(self.drop42(self.bn42(self.conv42(x41))))
        x43 = F.relu(self.drop43(self.bn43(self.conv43(x42))))
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)
        size4 = x43.size()

        # Stage 5
        x51 = F.relu(self.drop51(self.bn51(self.conv51(x4p))))
        x52 = F.relu(self.drop52(self.bn52(self.conv52(x51))))
        x53 = F.relu(self.drop53(self.bn53(self.conv53(x52))))
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)
        size5 = x53.size()

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=size5)
        x53d = F.relu(self.drop53d(self.bn53d(self.conv53d(x5d))))
        x52d = F.relu(self.drop52d(self.bn52d(self.conv52d(x53d))))
        x51d = F.relu(self.drop51d(self.bn51d(self.conv51d(x52d))))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2, output_size=size4)
        x43d = F.relu(self.drop43d(self.bn43d(self.conv43d(x4d))))
        x42d = F.relu(self.drop42d(self.bn42d(self.conv42d(x43d))))
        x41d = F.relu(self.drop41d(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2, output_size=size3)
        x33d = F.relu(self.drop33d(self.bn33d(self.conv33d(x3d))))
        x32d = F.relu(self.drop32d(self.bn32d(self.conv32d(x33d))))
        x31d = F.relu(self.drop31d(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2, output_size=size2)
        x22d = F.relu(self.drop22d(self.bn22d(self.conv22d(x2d))))
        x21d = F.relu(self.drop21d(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2, output_size=size1)
        x12d = F.relu(self.drop12d(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return x11d

class SegNetxxx(nn.Module):
    
    def __init__(self, num_class):
        super(SegNet, self).__init__()
        self.num_class = num_class
        
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)

        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        
        self.unpool5 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)
        self.unpool4 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)
        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)
        self.unpool2 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)
        self.unpool1 = nn.MaxUnpool2d(kernel_size = 2, stride = 2, padding = 0)

        self.deconv5_1 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.deconv5_2 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.deconv5_3 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.deconv4_1 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.deconv4_2 = nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.deconv4_3 = nn.ConvTranspose2d(512, 256, kernel_size = 3, stride = 1, padding = 1)
        self.deconv3_1 = nn.ConvTranspose2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.deconv3_2 = nn.ConvTranspose2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.deconv3_3 = nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 1, padding = 1)
        self.deconv2_1 = nn.ConvTranspose2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.deconv2_2 = nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 1, padding = 1)
        self.deconv1_1 = nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 1, padding = 1)
        self.deconv1_2 = nn.ConvTranspose2d(64, self.num_class, kernel_size = 3, stride = 1, padding = 1)

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
    
    def forward(self, x):
        
        size_1 = x.size()
        x = self.conv1_1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x, idxs1 = self.pool1(x)
        
        size_2 = x.size()
        x = self.conv2_1(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x, idxs2 = self.pool2(x)
        
        size_3 = x.size()
        x = self.conv3_1(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.conv3_3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x, idxs3 = self.pool3(x)
        
        size_4 = x.size()
        x = self.conv4_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv4_3(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x, idxs4 = self.pool4(x)
        
        size_5 = x.size()
        x = self.conv5_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv5_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x, idxs5 = self.pool5(x)

        
        x = self.unpool5(x, idxs5, output_size = size_5)
        x = self.deconv5_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv5_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv5_3(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        
        x = self.unpool4(x, idxs4, output_size = size_4)
        x = self.deconv4_1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv4_2(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.deconv4_3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        
        x = self.unpool3(x, idxs3, output_size = size_3)
        x = self.deconv3_1(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.deconv3_2(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.deconv3_3(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        
        x = self.unpool2(x, idxs2, output_size = size_2)
        x = self.deconv2_1(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.deconv2_2(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        
        x = self.unpool1(x, idxs1, output_size = size_1)
        x = self.deconv1_1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.deconv1_2(x)
        
        return x

import torch
from torch import nn
from torchvision import models

# from ..utils import initialize_weights
# from .config import vgg19_bn_path

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(_DecoderBlock, self).__init__()
        middle_channels = int(in_channels / 2)
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(middle_channels),
                      nn.ReLU(inplace=True),
                  ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class SegNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(SegNet, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        # if pretrained:
        #     vgg.load_state_dict(torch.load(vgg19_bn_path))
        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[0:7])
        self.enc2 = nn.Sequential(*features[7:14])
        self.enc3 = nn.Sequential(*features[14:27])
        self.enc4 = nn.Sequential(*features[27:40])
        self.enc5 = nn.Sequential(*features[40:])

        self.dec5 = nn.Sequential(
            *([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] +
              [nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               nn.ReLU(inplace=True)] * 4)
        )
        self.dec4 = _DecoderBlock(1024, 256, 4)
        self.dec3 = _DecoderBlock(512, 128, 4)
        self.dec2 = _DecoderBlock(256, 64, 2)
        self.dec1 = _DecoderBlock(128, num_classes, 2)
        initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.cat([enc4, dec5], 1))
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))
        return dec1
if __name__ == '__main__':  
    # net = \
    # {
    #     "model":"smallunet",
    #     "drop_rate":0.3,
    #     "bn_momentum": 0.1 
    # }
    # input = \
    # {
    #     "data_type": "float32",
    #     "matrix_size": [160,160],
    #     "resolution": "0.15x0.15",
    #     "orientation": "RAI"
    # }

    # net = SegNet(nb_input_channels=3, n_classes=1,
    #              mean=0, std=0,
    #              orientation= "RAI",
    #              resolution= "0.15x0.15" ,
    #              matrix_size= [160,160],
    #              drop_rate= 0.3,
    #              bn_momentum= 0.1)
    net = SegNet(num_class = 21)
    print(net)