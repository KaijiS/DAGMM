#coding:utf-8
#!/usr/bin/env python

import os
import argparse
import math

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from net import DAGMM


def train():
    parser = argparse.ArgumentParser(description='DAGMM')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', type=int, default=128, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of sweeps over the dataset to train')
    parser.add_argument('--cn_h_unit', type=int, default=10, help='Number of Compression Network hidden units')
    parser.add_argument('--cn_z_unit', type=int, default=2, help='Number of Compression Network z units')
    parser.add_argument('--en_h_unit', type=int, default=10 ,help='Number of Estimation Network hidden units')
    parser.add_argument('--en_o_unit', type=int, default=2 ,help='Number of Estimation Network output units')
    parser.add_argument('--out', '-o', default='result',help='Directory to output the result')
    parser.add_argument('--frequency', '-f', type=int, default=5, help='Frequency of taking a snapshot')
    parser.add_argument('--resume', '-r', type=int, help='Resume the training from snapshot that is designated epoch number')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# Compression Network: Dim - {0} - {1} - {0} - Dim'.format(args.cn_h_unit, args.cn_z_unit))
    print('# Estimation Network: {} - {} - {}'.format(args.cn_z_unit+2, args.en_h_unit, args.en_o_unit))
    print('# Output-directory: {}'.format(args.out))
    print('# Frequency-snapshot: {}'.format(args.frequency))
    if args.resume:
        print('# Resume-epochNumber: {}'.format(args.resume))
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

    # 型変換
    trainData = trainData.astype(np.float32)
    validData = validData.astype(np.float32)


    train_iter = chainer.iterators.SerialIterator(trainData, batch_size = args.batchsize, repeat=True, shuffle=True)
    valid_iter = chainer.iterators.SerialIterator(validData, batch_size = len(validData), repeat=False, shuffle=False)

    model = DAGMM(args.cn_h_unit, args.cn_z_unit, len(trainData[0]), args.en_h_unit, args.en_o_unit)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = optimizers.Adam(alpha=0.0001)
    optimizer.setup(model)

    if args.resume:
        serializers.load_npz(args.out + '/model_snapshot_epoch_'+str(args.resume), model)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu, loss_func=model.lossFunc(gpu=args.gpu))
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpu, eval_func=model.lossFunc(gpu=args.gpu)))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'), trigger=(args.frequency, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, filename='model_snapshot_epoch_{.updater.epoch}'), trigger=(args.frequency,'epoch'))
    trainer.extend(extensions.snapshot_object(optimizer, filename='optimizer_snapshot_epoch_{.updater.epoch}'), trigger=(args.frequency,'epoch'))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss1.png'))
    trainer.extend(extensions.PlotReport(['main/loss'], x_key='epoch', file_name='loss2.png'))
    trainer.extend(extensions.LogReport(log_name="log", trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        serializers.load_npz(args.out + '/snapshot_epoch-'+str(args.resume), trainer)

    trainer.run()

if __name__ == "__main__":
    train()
