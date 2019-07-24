#
# Inference
#

from __future__ import print_function
import os
import cv2
import numpy as np
from tqdm import tqdm
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

    _, _, transform_infer = transforms.get_transform(args.dataset)
    galleryset = datasets.get_dataset(args.dataset,
                                    root='/home/ace19/dl_data/materials/train',
                                    transform=transform_infer)
    queryset = datasets.get_dataset(args.dataset,
                                     split='eval',
                                     root='/home/ace19/dl_data/materials/query',
                                     transform=transform_infer)
    gallery_loader = DataLoader(
        galleryset, batch_size=args.batch_size, num_workers=args.workers)
    query_loader = torch.utils.data.DataLoader(
        queryset, batch_size=args.test_batch_size, num_workers=args.workers)

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


    gallery_features_list = []
    gallery_path_list = []
    query_features_list = []
    query_path_list = []
    def retrieval():
        model.eval()

        print(" ==> Load gallery ... ")
        tbar = tqdm(gallery_loader, desc='\r')
        for batch_idx, (gallery_paths, data, gt) in enumerate(tbar):
            if args.cuda:
                data, gt = data.cuda(), gt.cuda()

            with torch.no_grad():
                # output = model(data)
                # TTA
                batch_size, n_crops, c, h, w = data.size()
                # fuse batch size and ncrops
                output = model(data.view(-1, c, h, w))
                # avg over crops
                # output = output.view(batch_size, n_crops, -1).mean(1)
                features = output.get_features().view(batch_size, n_crops, -1).mean(1)
                gallery_features_list.extend(features)
                gallery_path_list.extend(gallery_paths)
        # end of for

        print(" ==> Load query ... ")
        tbar = tqdm(query_loader, desc='\r')
        for batch_idx, (query_paths, data) in enumerate(tbar):
            if args.cuda:
                data = data.cuda()

            with torch.no_grad():
                # TTA
                batch_size, n_crops, c, h, w = data.size()
                # fuse batch size and ncrops
                output = model(data.view(-1, c, h, w))
                # avg over crops
                # output = output.view(batch_size, n_crops, -1).mean(1)
                features = output.get_features().view(batch_size, n_crops, -1).mean(1)
                gallery_features_list.extend(features)
                query_features_list.extend(output.get_features())
                query_path_list.extend(query_paths)
        # end of for


        # # matching
        # top_indices = match(gallery_features_list, query_features_list)
        #
        # # display top 5 image correspond to target
        # display_retrieval(top_indices, gallery_path_list, query_path_list)

    retrieval()


if __name__ == "__main__":
    main()
