
from __future__ import print_function
import os
import cv2
import numpy as np
from tqdm import tqdm
import csv

import torch
import torch.nn as nn

import datasets
from models import model_zoo
import transforms
from option import Options


# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []


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
    _, _, transform_infer = transforms.get_transform(args.dataset)
    infer_set = datasets.get_dataset(args.dataset,
                                     split='eval',
                                     root='/home/ace19/dl_data/v2-plant-seedlings-dataset2/evalset',
                                     transform=transform_infer)
    infer_loader = torch.utils.data.DataLoader(
        infer_set, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # init the model
    model = model_zoo.get_model(args.model)
    print(model)

    if args.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)

    # check point
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            args.start_epoch = checkpoint['epoch'] + 1
            best_pred = checkpoint['best_pred']
            acclist_train = checkpoint['acclist_train']
            acclist_val = checkpoint['acclist_val']
            model.module.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.checkpoint, checkpoint['epoch']))
        else:
            raise RuntimeError("=> no infer checkpoint found at '{}'". \
                               format(args.checkpoint))
    else:
        raise RuntimeError("=> config \'args.checkpoint\' is '{}'". \
                           format(args.checkpoint))

    def eval():
        model.eval()

        submission = {}

        tbar = tqdm(infer_loader, desc='\r')
        for batch_idx, (name, data) in enumerate(tbar):
            if args.cuda:
                data = data.cuda()

            with torch.no_grad():
                # TTA
                batch_size, n_crops, c, h, w = data.size()
                # fuse batch size and ncrops
                output = model(data.view(-1, c, h, w))
                # avg over crops
                output = output.view(batch_size, n_crops, -1).mean(1)
                _, preds = torch.max(output, 1)

            size = len(name)
            for i in range(size):
                submission[name[i]] = preds[i].cpu()
        # end of for

        ########################
        # make submission.csv
        ########################
        if not os.path.exists('result'):
            os.makedirs('result')

        fout = open(os.path.join('result', args.result + '#20.csv'),
            'w', encoding='utf-8', newline='')
        writer = csv.writer(fout)
        # writer.writerow(['id', 'label'])
        for key in sorted(submission.keys()):
            name = key.split('/')[-1]
            writer.writerow([name, submission[key].numpy()])
        fout.close()

    eval()


if __name__ == "__main__":
    main()
