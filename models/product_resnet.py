
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.customize import *
from . import resnet
from datasets import datasets

__all__ = ['product_resnet50', 'product_resnet101', 'product_cosine_softmax']

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
            Normalize(),
            # nn.BatchNorm1d(2048),
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            Normalize(),
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



class Product_Cosine_Softmax(nn.Module):
    def __init__(self, nclass, pretrained, backbone, **kwargs):
        super(Product_Cosine_Softmax, self).__init__()
        self.nclass = nclass
        self.backbone = backbone
        # copying modules from pretrained models
        if self.backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained, dilated=False)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

        self.weights = torch.nn.Parameter(torch.randn(2048, self.nclass))
        self.scale = torch.nn.Parameter(F.softplus(torch.randn(())))
        # self.scale = F.softplus(self.scale)

        self.fc = nn.Linear(2048, 2048)


    def num_flat_features(self, x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_feature = 1
        for s in size:
            num_feature *= s

        return num_feature


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

        # cosine-softmax
        # feature_dim = x.size()[1]
        x = x.view(-1, self.num_flat_features(x))   # torch.Size([64, 2048])
        x = F.dropout2d(x, p=0.8)
        x = self.fc(x)

        features = x
        # Features in rows, normalize axis 1.
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        # weights = torch.randn(feature_dim, self.nclass).type(torch.FloatTensor)
        # scale = torch.randn(()).type(torch.FloatTensor)
        # self.scale = F.softplus(self.scale)

        # Mean vectors in colums, normalize axis 0.
        weights_normed = F.normalize(self.weights, p=2, dim=0, eps=1e-8)
        logits = self.scale.cuda() * torch.mm(features.cuda(), weights_normed.cuda())     # torch.matmul

        # return features, logits
        return logits


def product_resnet50(backbone_pretrained=False, **kwargs):
    model = Product_ResNet(datasets['product'.lower()].NUM_CLASS, backbone_pretrained, backbone='resnet50', **kwargs)
    return model


def product_resnet101(backbone_pretrained=False, **kwargs):
    model = Product_ResNet(datasets['product'.lower()].NUM_CLASS,  backbone_pretrained, backbone='resnet101', **kwargs)
    return model

def product_cosine_softmax(backbone_pretrained=False, **kwargs):
    model = Product_Cosine_Softmax(datasets['product'.lower()].NUM_CLASS,  backbone_pretrained, backbone='resnet101', **kwargs)
    return model
