
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.customize import *
from . import resnet
from datasets import datasets

from efficientnet_pytorch import EfficientNet


__all__ = ['product_resnet50', 'product_resnet101', 'product_cosine_softmax']

class Product_Net(nn.Module):
    def __init__(self, nclass, pretrained, backbone):
        super(Product_Net, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained model
        if self.backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained, dilated=False)
        elif self.backbone.startswith('efficientnet'):
            self.pretrained = EfficientNet.from_pretrained(self.backbone, num_classes=nclass)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

        in_channels = 512
        if self.backbone in ['resnet50', 'resnet101', 'resnet152']:
            in_channels = 2048
        elif self.backbone == 'efficientnet-b3':
            in_channels = 1536
        elif self.backbone == 'efficientnet-b4':
            in_channels = 1792

        self.head = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
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

        if self.backbone.startswith('efficientnet'):
            x = self.pretrained(x)  # torch.Size([batch_size, num_class])
            return x
            # features = self.pretrained.extract_features(x)
            # # print(features.shape)  # torch.Size([batch_size, 1536, 8, 50])
            # return self.head(features)
        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            x = self.pretrained.layer1(x)
            x = self.pretrained.layer2(x)
            x = self.pretrained.layer3(x)
            x = self.pretrained.layer4(x)
            # # print(x.shape)  # torch.Size([batch_size, 512, 8, 50]) - resnet34
            return self.head(x)



class Product_Cosine_Softmax(nn.Module):
    def __init__(self, nclass, pretrained, backbone, **kwargs):
        super(Product_Cosine_Softmax, self).__init__()
        self.nclass = nclass
        self.backbone = backbone
        # copying modules from pretrained model
        if self.backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained, dilated=False)
        elif self.backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained, dilated=False)
        elif self.backbone.startswith('efficientnet'):
            self.pretrained = EfficientNet.from_pretrained(self.backbone, num_classes=nclass)
        else:
            raise RuntimeError('unknown backbone: {}'.format(self.backbone))

        self.weights = torch.nn.Parameter(torch.randn(2048, self.nclass))
        self.scale = torch.nn.Parameter(F.softplus(torch.randn(())))

        self.fc = nn.Linear(2048, 2048)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def num_flat_features(self, x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_feature = 1
        for s in size:
            num_feature *= s

        return num_feature


    def forward(self, x):
        _, _, h, w = x.size()
        if self.backbone.startswith('efficientnet'):
            x = self.pretrained.extract_features(x)
            # print(features.shape)  # torch.Size([batch_size, 1536, 10, 10])
        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            x = self.pretrained.layer1(x)
            x = self.pretrained.layer2(x)
            x = self.pretrained.layer3(x)
            x = self.pretrained.layer4(x)

        x = self.avgpool(x)

        # cosine-softmax
        # feature_dim = x.size()[1]
        x = x.view(-1, self.num_flat_features(x))   # torch.Size([64, 2048])
        x = F.dropout2d(x, p=0.8)
        x = self.fc(x)

        features = x
        # Features in rows, normalize axis 1.
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        # Mean vectors in colums, normalize axis 0.
        weights_normed = F.normalize(self.weights, p=2, dim=0, eps=1e-8)
        logits = self.scale.cuda() * torch.mm(features.cuda(), weights_normed.cuda())     # torch.matmul

        return features, logits


def product_resnet50(backbone_pretrained=False, **kwargs):
    model = Product_Net(datasets['product'.lower()].NUM_CLASS, backbone_pretrained, **kwargs)
    return model


def product_resnet101(backbone_pretrained=False, **kwargs):
    model = Product_Net(datasets['product'.lower()].NUM_CLASS,  backbone_pretrained, **kwargs)
    return model


def product_cosine_softmax(backbone_pretrained=False, **kwargs):
    model = Product_Cosine_Softmax(datasets['product'.lower()].NUM_CLASS,  backbone_pretrained, **kwargs)
    return model
