
from __future__ import print_function
import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import datasets
from models import model_zoo
import transforms
from utils import lr_scheduler, files
from option import Options
from datasets.sampler import ImbalancedDatasetSampler

# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []

# Writer will output to ./runs/ directory by default
# writer = SummaryWriter()


def main():
    # init the args
    global best_pred, acclist_train, acclist_val
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # init dataloader
    transform_train, transform_val, _ = transforms.get_transform(args.dataset)
    trainset = datasets.get_dataset(args.dataset,
                                    root='/home/ace19/dl_data/materials/train',
                                    transform=transform_train)
    valset = datasets.get_dataset(args.dataset,
                                  root='/home/ace19/dl_data/materials/validation',
                                  transform=transform_val)

    # balanced sampling between classes
    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, num_workers=args.workers,
        sampler=ImbalancedDatasetSampler(trainset))
    # train_loader = DataLoader(
    #     trainset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(
        valset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers)

    # init the backbone model
    if args.pretrained is not None:
        model = model_zoo.get_model(args.model)
    else:
        model = model_zoo.get_model(args.model, backbone_pretrained=True)
    print(model)
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
    #                             weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        criterion.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)
    # check point
    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            args.start_epoch = checkpoint['epoch'] + 1
            best_pred = checkpoint['best_pred']
            acclist_train = checkpoint['acclist_train']
            acclist_val = checkpoint['acclist_val']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pretrained, checkpoint['epoch']))
        else:
            raise RuntimeError("=> no pretrained checkpoint found at '{}'". \
                               format(args.pretrained))

    scheduler = lr_scheduler.LR_Scheduler(args.lr_scheduler, args.lr, args.epochs,
                                            len(train_loader), args.lr_step)

    def train(epoch):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()

        global best_pred, acclist_train, acclist_val

        tbar = tqdm(train_loader, desc='\r')
        for batch_idx, (_, data, target) in enumerate(tbar):
            scheduler(optimizer, batch_idx, epoch, best_pred)
            # display_data(data)
            # TODO: Convert from list of 3D to 4D
            # data = np.stack(data, axis=1)
            # data = torch.from_numpy(data)

            if args.cuda:
                data, target = data.cuda(), target.cuda()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            _, output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            acc1 = accuracy(output, target)
            top1.update(acc1[0], data.size(0))
            losses.update(loss.item(), data.size(0))
            tbar.set_description('\rLoss: %.3f | Top1: %.3f' % (losses.avg, top1.avg))

        acclist_train += [top1.avg]


    def validate(epoch):
        model.eval()
        top1 = AverageMeter()
        top5 = AverageMeter()
        confusion_matrix = torch.zeros(args.nclass, args.nclass)

        global best_pred, acclist_train, acclist_val
        is_best = False

        tbar = tqdm(val_loader, desc='\r')
        # TTA(TenCrop) input, target = batch # input is a 5d tensor, target is 2d
        # bs, ncrops, c, h, w = input.size()
        # result = model(input.view(-1, c, h, w))  # fuse batch size and ncrops
        # result_avg = result.view(bs, ncrops, -1).mean(1)  # avg over crops
        for batch_idx, (name, data, target) in enumerate(tbar):
            # Convert from list of 3D to 4D
            # data = np.stack(data, axis=1)
            # data = torch.from_numpy(data)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
                # data, target = Variable(data), Variable(target)
            with torch.no_grad():
                # _, output = model(data)

                # TTA
                batch_size, n_crops, c, h, w = data.size()
                # fuse batch size and ncrops
                _, output = model(data.view(-1, c, h, w))
                # avg over crops
                output = output.view(batch_size, n_crops, -1).mean(1)
                # accuracy
                acc1, acc5 = accuracy(output, target, topk=(1, 1))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))

                # confusion matrix
                _, preds = torch.max(output, 1)
                for t, p in zip(target.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            tbar.set_description('Top1: %.3f | Top5: %.3f' % (top1.avg, top5.avg))
        # end of for

        print('\n----------------------------------')
        print('confusion matrix:\n', confusion_matrix)
        # get the per-class accuracy
        print('\nper-class accuracy(precision):\n', confusion_matrix.diag() / confusion_matrix.sum(1))
        print('----------------------------------\n')

        if args.eval:
            print('Top1 Acc: %.3f | Top5 Acc: %.3f ' % (top1.avg, top5.avg))
            return

        # save checkpoint
        acclist_val += [top1.avg]
        if top1.avg > best_pred:
            best_pred = top1.avg
            is_best = True
        files.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'acclist_train': acclist_train,
            'acclist_val': acclist_val,
        }, args=args, is_best=is_best)

    if args.eval:
        validate(args.start_epoch)
        # writer.close()
        return

    for epoch in range(args.start_epoch, args.epochs + 1):
        train(epoch)
        validate(epoch)
        # writer.close()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        prob, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def display_data(data):
    # display image to verify
    data = data.numpy()
    data = np.transpose(data, (0, 2, 3, 1))
    # # assets not np.any(np.isnan(data))
    n_batch = data.shape[0]
    # n_view = train_batch_xs.shape[1]
    for i in range(n_batch):
        img = data[i]
        # scipy.misc.toimage(img).show() Or
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imwrite('/home/ace19/Pictures/' + str(i) + '.png', img)
        # cv2.imshow(str(train_batch_ys[idx]), img)
        cv2.waitKey(100)
        cv2.destroyAllWindows()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()
