from __future__ import print_function

import os
import cv2
import numpy as np
from tqdm import tqdm
import csv

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets
from models import model_zoo
import transforms
from option import Options

import retrieval.matching as matching


# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []


TOP_N = 5
RESULT_PATH = '/home/ace19/dl_result/image_retrieve/_result'


def _print_distances(distance_matrix, top_n_indice):
    distances = []
    num_row, num_col = top_n_indice.shape
    for r in range(num_row):
        col = []
        for c in range(num_col):
            col.append(distance_matrix[r, top_n_indice[r,c]])
        distances.append(col)

    return distances


def match_n(top_n, galleries, queries):
    # The distance metric used for measurement to query.
    metric = matching.NearestNeighborDistanceMetric("cosine")
    distance_matrix = metric.distance(queries, galleries)

    # top_indice = np.argmin(distance_matrix, axis=1)
    # top_n_indice = np.argpartition(distance_matrix, top_n, axis=1)[:, :top_n]
    # top_n_dist = _print_distances(distance_matrix, top_n_indice)
    # top_n_indice2 = np.argsort(top_n_dist, axis=1)
    # dist2 = _print_distances(distance_matrix, top_n_indice2)

    # TODO: need improvement.
    top_n_indice = np.argsort(distance_matrix, axis=1)[:, :top_n]
    top_n_distance = _print_distances(distance_matrix, top_n_indice)

    return top_n_indice, top_n_distance


def show_retrieval_result(top_n_indice, top_n_distance, gallery_path_list, query_path_list):
    col = top_n_indice.shape[1]
    for row_idx, query_img_path in enumerate(query_path_list):
        fig, axes = plt.subplots(ncols=6, figsize=(300, 300))
        axes[0].imshow(Image.open(query_img_path))

        for i in range(col):
            axes[i+1].imshow(Image.open(gallery_path_list[top_n_indice[row_idx, i]]))
        # plt.show()
        fig.savefig(os.path.join(RESULT_PATH, query_img_path.split('/')[-1]))
        plt.close()



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
                                    root='/home/ace19/dl_data/materials/gallery',
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

        print(" ==> Loading gallery ... ")
        tbar = tqdm(gallery_loader, desc='\r')
        for batch_idx, (gallery_paths, data, gt) in enumerate(tbar):
            if args.cuda:
                data, gt = data.cuda(), gt.cuda()

            with torch.no_grad():
                # features [256, 2048]
                # output [256, 128]
                # features, output = model(data)

                # TTA
                batch_size, n_crops, c, h, w = data.size()
                # fuse batch size and ncrops
                features, _ = model(data.view(-1, c, h, w))
                # avg over crops
                features = features.view(batch_size, n_crops, -1).mean(1)
                gallery_features_list.extend(features)
                gallery_path_list.extend(gallery_paths)
        # end of for

        print(" ==> Loading query ... ")
        tbar = tqdm(query_loader, desc='\r')
        for batch_idx, (query_paths, data) in enumerate(tbar):
            if args.cuda:
                data = data.cuda()

            with torch.no_grad():
                # TTA
                batch_size, n_crops, c, h, w = data.size()
                # fuse batch size and ncrops
                features, _ = model(data.view(-1, c, h, w))
                # avg over crops
                features = features.view(batch_size, n_crops, -1).mean(1)
                query_features_list.extend(features)
                query_path_list.extend(query_paths)
        # end of for

        # # matching
        top_n_indice, top_n_distance = \
            match_n(TOP_N,
                    torch.stack(gallery_features_list).cpu(),
                    torch.stack(query_features_list).cpu())

        # Show n images from the gallery similar to the query image.
        show_retrieval_result(top_n_indice, top_n_distance, gallery_path_list, query_path_list)

    retrieval()


if __name__ == "__main__":
    main()
