
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.customize import *
from . import resnet
from datasets import datasets

__all__ = ['product_resnet50', 'product_resnet101']

class Product_ResNet(nn.Module):
    def __init__(self, nclass, pretrained, backbone):
        super(Product_ResNet, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        if self.backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained, dilated=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

        self.head = nn.Sequential(
            View(-1, 2048),
            # Normalize(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, nclass),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        x = self.pretrained.avgpool(x)
        return self.head(x)


def product_resnet50(backbone_pretrained=False, **kwargs):
    model = Product_ResNet(datasets['product'.lower()].NUM_CLASS, backbone_pretrained, backbone='resnet50', **kwargs)
    return model


def product_resnet101(backbone_pretrained=False, **kwargs):
    model = Product_ResNet(datasets['product'.lower()].NUM_CLASS,  backbone_pretrained, backbone='resnet101', **kwargs)
    return model
