import torch
from torchvision.transforms import *

def get_transform(dataset, large_test_crop=False):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

    if dataset == 'product':
        transform_train = Compose([
            Resize(224),
            RandomResizedCrop(200),
            # # CenterCrop(320),
            RandomHorizontalFlip(),
            # ColorJitter(0.3, 0.3, 0.3),
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(contrast=0.5),
                transforms.ColorJitter(saturation=0.5),
                transforms.ColorJitter(hue=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            ]),
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomRotation((90, 90)),
                ]),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomRotation((270, 270)),
                ])
            ]),
            ToTensor(),
            Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
            normalize,
        ])
        transform_val = Compose([
            # Resize(180),
            # CenterCrop(180),
            # ToTensor(),
            # normalize,

            Resize(224),
            # this is a list of PIL Images
            TenCrop(200),
            # returns a 4D tensor
            Lambda(lambda crops: torch.stack([normalize(ToTensor()(crop)) for crop in crops])),
        ])
        transform_test = Compose([
            # Resize(180),
            # CenterCrop(180),
            # ToTensor(),
            # normalize,

            Resize(224),
            # this is a list of PIL Images
            TenCrop(200),
            # returns a 4D tensor
            Lambda(lambda crops: torch.stack([normalize(ToTensor()(crop)) for crop in crops])),
        ])
    return transform_train, transform_val, transform_test

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
