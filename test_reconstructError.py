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
    _, _, _, y_htr = model.fwd(trainData)
    # print(y_htr.data)
    print('# mse: {}'.format(F.mean_squared_error(trainData, y_htr)))
    se_htr = F.sum(F.squared_error(trainData, y_htr),axis=1)
    euc_htr = model.relativeEuclideanDistance(trainData, y_htr)
    cos_htr = model.cosineSimilarity(trainData, y_htr)


    print("------------------")
    print("Health testData Energy")
    _, _, _, y_hte = model.fwd(validData)
    # print(y_hte.data)
    print('# mse: {}'.format(F.mean_squared_error(validData, y_hte)))
    se_hte = F.sum(F.squared_error(validData, y_hte),axis=1)
    euc_hte = model.relativeEuclideanDistance(validData, y_hte)
    cos_hte = model.cosineSimilarity(validData, y_hte)

    print("------------------")
    print("Disease testData Energy")
    _, _, _, y_di = model.fwd(diseaseData)
    # print(y_di.data)
    print('# mse: {}'.format(F.mean_squared_error(diseaseData, y_di)))
    se_di = F.sum(F.squared_error(diseaseData, y_di),axis=1)
    euc_di = model.relativeEuclideanDistance(diseaseData, y_di)
    cos_di = model.cosineSimilarity(diseaseData, y_di)
    print("")

    print("------------------")
    print("squared_error")
    plt.hist(se_htr.data, bins=100, alpha=0.4, histtype='stepfilled', color='b')
    plt.hist(se_hte.data, bins=100, alpha=0.4, histtype='stepfilled', color='g')
    plt.hist(se_di.data, bins=100, alpha=0.4, histtype='stepfilled', color='r')
    plt.show()

    print("------------------")
    print("relativeEuclideanDistance")
    plt.hist(euc_htr.data, bins=100, alpha=0.4, histtype='stepfilled', color='b')
    plt.hist(euc_hte.data, bins=100, alpha=0.4, histtype='stepfilled', color='g')
    plt.hist(euc_di.data, bins=100, alpha=0.4, histtype='stepfilled', color='r')
    plt.show()

    print("------------------")
    print("cosineSimilarity")
    plt.hist(cos_htr.data, bins=100, alpha=0.4, histtype='stepfilled', color='b')
    plt.hist(cos_hte.data, bins=100, alpha=0.4, histtype='stepfilled', color='g')
    plt.hist(cos_di.data, bins=100, alpha=0.4, histtype='stepfilled', color='r')
    plt.show()


if __name__ == "__main__":
    test()
