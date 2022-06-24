import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import gc
from copy import deepcopy
import random



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    A wrapup of a residual block
    """
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ImageResNet(nn.Module):
    """
    The image embedder that is implemented as a residual network 
    """

    def __init__(self, block, layers, output_channels=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ImageResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1
        # if replace_stride_with_dilation is None:
        #     # each element in the tuple indicates if we should replace
        #     # the 2x2 stride with a dilated convolution instead
        #     replace_stride_with_dilation = [False, False, False]
        # if len(replace_stride_with_dilation) != 3:
        #     raise ValueError("replace_stride_with_dilation should be None "
        #                      "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 128, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 256, layers[5], stride=2)
        self.layer7 = self._make_layer(block, 512, layers[6], stride=2)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc1 = nn.Linear(2048, output_channels)
        # self.fc1 = nn.Linear(768, 24)
        # self.bn2 = nn.BatchNorm1d(24)
        # self.fc2 = nn.Linear(24, output_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, 
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, use_lowerlevel_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # batch_size, 8, 512, 512
        # x = self.maxpool(x)

        x = self.layer1(x) # batch_size, 8, 256, 256
        x = self.layer2(x) # batch_size, 16, 128, 128
        x = self.layer3(x) # batch_size, 32, 64, 64
        x = self.layer4(x) # batch_size, 64, 32, 32
        x = self.layer5(x) # batch_size, 128, 16, 16
        x = self.layer6(x) # batch_size, 192, 8, 8
        lowerlevel_img_feat = x 
        x = self.layer7(x) # batch_size, 192, 4, 4
        
        x = self.avgpool(x)
        z = torch.flatten(x, 1) # batch_size, 768
        # x = self.fc1(z)
        # x = self.bn2(x)
        # x = self.relu(x)
        # y = self.fc2(x)
        outputs = (z,)
        logits = self.fc1(z)
        outputs += (logits,)
        if use_lowerlevel_features:
            outputs += (lowerlevel_img_feat,)
        return outputs # z, (logits), (layer6 output)

    
class StudentResNet(nn.Module):
    """
    The image embedder that is implemented as a residual network 
    """

    def __init__(self, block, layers, output_channels=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(StudentResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1
        # if replace_stride_with_dilation is None:
        #     # each element in the tuple indicates if we should replace
        #     # the 2x2 stride with a dilated convolution instead
        #     replace_stride_with_dilation = [False, False, False]
        # if len(replace_stride_with_dilation) != 3:
        #     raise ValueError("replace_stride_with_dilation should be None "
        #                      "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 64, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 192, layers[5], stride=2)
        self.layer7 = self._make_layer(block, 192, layers[6], stride=2)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc1 = nn.Linear(768, output_channels)
        # self.fc1 = nn.Linear(768, 24)
        # self.bn2 = nn.BatchNorm1d(24)
        # self.fc2 = nn.Linear(24, output_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, 
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, use_lowerlevel_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # batch_size, 8, 512, 512
        # x = self.maxpool(x)

        x = self.layer1(x) # batch_size, 8, 256, 256
        x = self.layer2(x) # batch_size, 16, 128, 128
        x = self.layer3(x) # batch_size, 32, 64, 64
        x = self.layer4(x) # batch_size, 64, 32, 32
        x = self.layer5(x) # batch_size, 128, 16, 16
        x = self.layer6(x) # batch_size, 192, 8, 8
        lowerlevel_img_feat = x 
        x = self.layer7(x) # batch_size, 192, 4, 4
        
        x = self.avgpool(x)
        z = torch.flatten(x, 1) # batch_size, 768
        # x = self.fc1(z)
        # x = self.bn2(x)
        # x = self.relu(x)
        # y = self.fc2(x)
        outputs = (z,)
        logits = self.fc1(z)
        outputs += (logits,)
        if use_lowerlevel_features:
            outputs += (lowerlevel_img_feat,)
        return outputs # z, (logits), (layer6 output)

def mixup_data(x, y, lam):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
   
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    if torch.cuda.is_available():
        index = index.cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


class ImageResNetMixup(nn.Module):
    """
    The image embedder that is implemented as a residual network 
    """

    def __init__(self, block, layers, output_channels=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ImageResNetMixup, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1
        # if replace_stride_with_dilation is None:
        #     # each element in the tuple indicates if we should replace
        #     # the 2x2 stride with a dilated convolution instead
        #     replace_stride_with_dilation = [False, False, False]
        # if len(replace_stride_with_dilation) != 3:
        #     raise ValueError("replace_stride_with_dilation should be None "
        #                      "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 128, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 256, layers[5], stride=2)
        self.layer7 = self._make_layer(block, 512, layers[6], stride=2)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc1 = nn.Linear(2048, output_channels)
        # self.fc1 = nn.Linear(768, 24)
        # self.bn2 = nn.BatchNorm1d(24)
        # self.fc2 = nn.Linear(24, output_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, 
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, target= None, mixup=False, mixup_hidden=True, mixup_alpha=None , lam = 0.4):

#     def forward(self, x, use_lowerlevel_features=False):
        if target is not None: 
            if mixup_hidden:
                layer_mix = random.randint(0,3)
            elif mixup:
                layer_mix = 0
            else:
                layer_mix = None   

            out = x

            target_a = target_b  = target

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x) # batch_size, 8, 512, 512
            # x = self.maxpool(x)

            x = self.layer1(x) # batch_size, 8, 256, 256
            if layer_mix == 0:
                x, target_a , target_b , lam = mixup_data(x, target, lam=lam)

            x = self.layer2(x) # batch_size, 16, 128, 128
            if layer_mix == 1:
                x, target_a , target_b , lam = mixup_data(x, target, lam=lam)

            x = self.layer3(x) # batch_size, 32, 64, 64
            if layer_mix == 2:
                x, target_a , target_b , lam = mixup_data(x, target, lam=lam)

            x = self.layer4(x) # batch_size, 64, 32, 32
            if layer_mix == 3:
                x, target_a , target_b , lam = mixup_data(x, target, lam=lam)

            x = self.layer5(x) # batch_size, 128, 16, 16
            x = self.layer6(x) # batch_size, 192, 8, 8
            lowerlevel_img_feat = x 
            x = self.layer7(x) # batch_size, 192, 4, 4

            x = self.avgpool(x)
            z = torch.flatten(x, 1) # batch_size, 768
            # x = self.fc1(z)
            # x = self.bn2(x)
            # x = self.relu(x)
            # y = self.fc2(x)
            outputs = (z,)
            logits = self.fc1(z)
            outputs += (logits,)
#             if use_lowerlevel_features:
#                 outputs += (lowerlevel_img_feat,)
            return z , logits , target_a , target_b
#             return outputs
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x) # batch_size, 8, 512, 512
            # x = self.maxpool(x)

            x = self.layer1(x) # batch_size, 8, 256, 256
            x = self.layer2(x) # batch_size, 16, 128, 128
            x = self.layer3(x) # batch_size, 32, 64, 64
            x = self.layer4(x) # batch_size, 64, 32, 32
            x = self.layer5(x) # batch_size, 128, 16, 16
            x = self.layer6(x) # batch_size, 192, 8, 8
            lowerlevel_img_feat = x 
            x = self.layer7(x) # batch_size, 192, 4, 4

            x = self.avgpool(x)
            z = torch.flatten(x, 1) # batch_size, 768
            # x = self.fc1(z)
            # x = self.bn2(x)
            # x = self.relu(x)
            # y = self.fc2(x)
            outputs = (z,)
            logits = self.fc1(z)
            outputs += (logits,)
#             if use_lowerlevel_features:
#                 outputs += (lowerlevel_img_feat,)
            return outputs # z, (logits), (layer6 output)

from torch.nn.utils.weight_norm import WeightNorm

# Basic ResNet model

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
#         print(x_norm.shape)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return None, scores