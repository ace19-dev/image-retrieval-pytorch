import torch
import numpy as np

from torch.utils.data.sampler import WeightedRandomSampler


numDataPoints = 1000
data_dim = 5
bs = 100

data = torch.FloatTensor(numDataPoints, data_dim)
target = np.hstack((np.zeros(int(numDataPoints * 0.9), dtype=np.int32),
                    np.ones(int(numDataPoints * 0.1), dtype=np.int32)))

dataset_x = data.numpy()
dataset_y = target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset_x,
                                                    dataset_y,
                                                    test_size=0.33,
                                                    random_state=42,
                                                    stratify = dataset_y)
print('target train: {}/{}'.format(len(np.where(y_train==0)[0]),
                                   len(np.where(y_train==1)[0])))

class_sample_count = np.array([len(np.where(y_train==t)[0]) for t in np.unique(y_train)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_train])

samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train.astype(int)))
validDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test.astype(int)))

trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=bs, num_workers=1, sampler = sampler)
testLoader = torch.utils.data.DataLoader(dataset = validDataset, batch_size=bs, shuffle=False, num_workers=1)

for i, (data, target) in enumerate(trainLoader):
    print("batch index {}, 0/1: {}/{}".format(
        i, len(np.where(target.numpy()==0)[0]), len(np.where(target.numpy()==1)[0])))