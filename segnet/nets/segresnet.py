import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from itertools import chain
from math import ceil
from itertools import chain
from segnet.utils.utils import set_trainable

class DecoderBottleneck(nn.Module):
    def __init__(self, inchannels):
        super(DecoderBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels//4)
        self.conv2 = nn.ConvTranspose2d(inchannels//4, inchannels//4, kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels//4)
        self.conv3 = nn.Conv2d(inchannels//4, inchannels//2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels//2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.ConvTranspose2d(inchannels, inchannels//2, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(inchannels//2))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class LastBottleneck(nn.Module):
    def __init__(self, inchannels):
        super(LastBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels//4)
        self.conv2 = nn.Conv2d(inchannels//4, inchannels//4, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels//4)
        self.conv3 = nn.Conv2d(inchannels//4, inchannels//4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels//4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
                nn.Conv2d(inchannels, inchannels//4, kernel_size=1, bias=False),
                nn.BatchNorm2d(inchannels//4))
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SegResNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, pretrained=True, freeze_bn=False, **_):
        super(SegResNet, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        encoder = list(resnet50.children())
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        encoder[3].return_indices = True

        # Encoder
        self.first_conv = nn.Sequential(*encoder[:4])
        resnet50_blocks = list(resnet50.children())[4:-2]
        self.encoder = nn.Sequential(*resnet50_blocks)

        # Decoder
        resnet50_untrained = models.resnet50(pretrained=False)
        resnet50_blocks = list(resnet50_untrained.children())[4:-2][::-1]
        decoder = []
        channels = (2048, 1024, 512)
        for i, block in enumerate(resnet50_blocks[:-1]):
            new_block = list(block.children())[::-1][:-1]
            decoder.append(nn.Sequential(*new_block, DecoderBottleneck(channels[i])))
        new_block = list(resnet50_blocks[-1].children())[::-1][:-1]
        decoder.append(nn.Sequential(*new_block, LastBottleneck(256)))

        self.decoder = nn.Sequential(*decoder)
        self.last_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        )
        if freeze_bn: self.freeze_bn()
       

    def forward(self, x):
        inputsize = x.size()

        # Encoder
        x, indices = self.first_conv(x)
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)
        h_diff = ceil((x.size()[2] - indices.size()[2]) / 2)
        w_diff = ceil((x.size()[3] - indices.size()[3]) / 2)
        if indices.size()[2] % 2 == 1:
            x = x[:, :, h_diff:x.size()[2]-(h_diff-1), w_diff: x.size()[3]-(w_diff-1)]
        else:
            x = x[:, :, h_diff:x.size()[2]-h_diff, w_diff: x.size()[3]-w_diff]

        x = F.max_unpool2d(x, indices, kernel_size=2, stride=2)
        x = self.last_conv(x)
        
        if inputsize != x.size():
            h_diff = (x.size()[2] - inputsize[2]) // 2
            w_diff = (x.size()[3] - inputsize[3]) // 2
            x = x[:, :, h_diff:x.size()[2]-h_diff, w_diff: x.size()[3]-w_diff]
            if h_diff % 2 != 0: x = x[:, :, :-1, :]
            if w_diff % 2 != 0: x = x[:, :, :, :-1]

        return x

    def get_backbone_params(self):
        return chain(self.first_conv.parameters(), self.encoder.parameters())

    def get_decoder_params(self):
        return chain(self.decoder.parameters(), self.last_conv.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

    def freeze_backbone(self):
        set_trainable([self.first_conv, self.encoder], False)

    def unfreeze_backbone(self):
        set_trainable([self.first_conv, self.encoder], True)