import os
from PIL import Image

import torch
import torch.utils.data as data

class ProductDataset(data.Dataset):
    NUM_CLASS = 12
    def __init__(self, root, split='train', transform=None):
        self.split = split
        self.transform = transform

        if split == 'eval':
            self.images = make_test_dataset(root)
        else:
            classes, class_to_idx = find_classes(root)
            self.images, self.labels = make_dataset(root, class_to_idx)
            assert (len(self.images) == len(self.labels))


    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            _img = self.transform(_img)

        if self.split == 'eval':
            return self.images[index], _img
        else:
            _label = self.labels[index]
            return self.images[index], _img, _label


    def __len__(self):
        return len(self.images)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_test_dataset(datadir):
    images = []

    img_list = os.listdir(datadir)
    img_list.sort()
    for img in img_list:
        img_path = os.path.join(datadir, img)
        images.append(img_path)

    return images


def make_dataset(datadir, class_to_idx):
    images = []
    labels = []

    classes = os.listdir(datadir)
    for label in classes:
        img_path = os.path.join(datadir, label)
        _images = os.listdir(img_path)
        for img in _images:
            _image = os.path.join(img_path, img)
            images.append(_image)
            labels.append(class_to_idx[label])

    return images, labels

