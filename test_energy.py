#coding:utf-8
#!/usr/bin/env python

import argparse
import math

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from net import DAGMM

from matplotlib import pylab as plt


def test():
    parser = argparse.ArgumentParser(description='DAGMM')
    parser.add_argument('--epoch', '-e', type=int, default=10000, help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',help='Directory to output the result')
    parser.add_argument('--cn_h_unit', type=int, default=10, help='Number of Compression Network hidden units')
    parser.add_argument('--cn_z_unit', type=int, default=2, help='Number of Compression Network z units')
    parser.add_argument('--en_h_unit', type=int, default=10 ,help='Number of Estimation Network hidden units')
    parser.add_argument('--en_o_unit', type=int, default=2 ,help='Number of Estimation Network output units')
    args = parser.parse_args()

    print('# epoch: {}'.format(args.epoch))
    print('# Output-directory when training: {}'.format(args.out))
    print('# Compression Network: Dim - {0} - {1} - {0} - Dim'.format(args.cn_h_unit, args.cn_z_unit))
    print('# Estimation Network: {} - {} - {}'.format(args.cn_z_unit+2, args.en_h_unit, args.en_o_unit))
    print('')

    # データセット読み込み
    x_data = np.loadtxt('./dataset_arrhythmia/ExplanatoryVariables.csv',delimiter=',')
    y_label = np.loadtxt('./dataset_arrhythmia/CriterionVariables.csv',delimiter=',')

    # 正常データのみを抽出
    HealthData = x_data[y_label[:]==1]

    # 正常データを学習用と検証用に分割
    NumOfHealthData = len(HealthData)
    trainData = HealthData[:math.floor(NumOfHealthData*0.9)]
    validData = HealthData[len(trainData):]

    # 正常ではないデータ(異常データ)を抽出
    diseaseData = x_data[y_label[:]!=1]

    # 型変換
    trainData = trainData.astype(np.float32)
    validData = validData.astype(np.float32)
    diseaseData = diseaseData.astype(np.float32)

    model = DAGMM(args.cn_h_unit, args.cn_z_unit, len(trainData[0]), args.en_h_unit, args.en_o_unit)
    optimizer = optimizers.Adam(alpha=0.0001)
    optimizer.setup(model)


    print("------------------")
    print("Health trainData Energy")
    _, energy_htr, _, _ = model.fwd(trainData)
    # print(energy_htr.data)

    print("------------------")
    print("Health testData Energy")
    _, energy_hte, _, _ = model.fwd(validData)
    # print(energy_hte.data)

    print("------------------")
    print("Disease testData Energy")
    _, energy_di, _, _ = model.fwd(diseaseData)
    # print(energy_di.data)

    plt.hist(energy_htr.data, bins=100, alpha=0.4, histtype='stepfilled', color='b')
    plt.hist(energy_hte.data, bins=100, alpha=0.4, histtype='stepfilled', color='g')
    plt.hist(energy_di.data, bins=100, alpha=0.4, histtype='stepfilled', color='r')
    plt.show()


if __name__ == "__main__":
    test()
