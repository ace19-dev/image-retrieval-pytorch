import argparse
import os

class Options():
    def __init__(self):
        # Training settings
        parser = argparse.ArgumentParser(description='Image Retrieval')
        parser.add_argument('--dataset', type=str, default='product',
            help='training datasets (default: cifar10)')
        # model params
        parser.add_argument('--model', type=str, default='product_cosine_softmax',
            help='network model type (default: densenet)')
        parser.add_argument('--pretrained', type=str,
                            default=None,
                            # default='pre-trained/model_best.pth.tar',
                            help='load pretrianed mode')
        parser.add_argument('--nclass', type=int, default=30, metavar='N',
            help='number of classes (default: 10)')
        # training hyper params
        parser.add_argument('--batch-size', type=int, default=256,
            metavar='N', help='batch size for training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=256,
            metavar='N', help='batch size for testing (default: 256)')
        parser.add_argument('--epochs', type=int, default=100, metavar='N',
            help='number of epochs to train (default: 600)')
        parser.add_argument('--start_epoch', type=int, default=1,
            metavar='N', help='the epoch number to start (default: 1)')
        parser.add_argument('--workers', type=int, default=16,
            metavar='N', help='dataloader threads')
        # lr setting
        parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
            help='learning rate (default: 0.1)')
        parser.add_argument('--lr-scheduler', type=str, default='cos',
            help='learning rate scheduler (default: cos)')
        parser.add_argument('--lr-step', type=int, default=80, metavar='LR',
            help='learning rate step (default: 40)')
        # optimizer
        parser.add_argument('--momentum', type=float, default=0.9,
            metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=5e-4,
            metavar ='M', help='SGD weight decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true',
            default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--checkpoint', type=str,
                            # default=None,
                            default='../runs/product/product_cosine_softmax/default/model_best.pth.tar',
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
            help='set the checkpoint name')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
            help='evaluating')
        parser.add_argument('--result', type=str, default='test_results',
                            help='result file name')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args
