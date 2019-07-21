'''
데이터 분포 살펴보기

1. 기술 통계 - 데이터의 분포의 특징을 대표하는 값
    데이터의 숫자 (count)
    평균 (mean, average)
    분산 (variance)
    표준 편차 (standard deviation)
    최댓값 (maximum)
    최솟값 (minimum)
    중앙값 (median)
    사분위수 (quartile)

2. 히스토그램 - 히스토그램은 자료 값이 가질 수 있는 범위를 몇 개의 구간으로 나누고
    각 구간에 해당하는 값의 숫자 혹은 상대적 빈도를 계산하는 방법이다.

3. 커널 밀도 - 커널 밀도는 커널이라고 하는 특정 구간의 분포를 묘사하는 함수의 집합을 사용하여 전체 분포를 묘사하는 방법이다.
    커널 밀도를 사용하면 분포의 전체 모양을 파악하기가 더 쉽다.

TODO: train/val/test dataset 전체에 대한 눈에 보이는 통계가 필요.

'''


import os
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd


def main(args):
    dataset = os.listdir(args.dataset_dir)
    dataset.sort()
    for img in dataset:
        path = os.path.join(args.dataset_dir, img)
        img = cv2.imread(path, 0)
        s = pd.Series(img.ravel())
        print('image name: ', path.split('/')[-1], '\n', s.describe())
        print('--------------------------------------------')
        plt.hist(img.ravel(), 256, [0, 256])
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='/home/ace19/dl_data/dlp_competition/train/negative',
        help='Where is image to load'
    )

    args = parser.parse_args()
    main(args)