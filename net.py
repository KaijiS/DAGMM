#coding:utf-8
#!/usr/bin/env python

import os
import math
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class DAGMM(Chain):
    def __init__(self, CN_h_units, CN_z_units, CN_o_units, EN_h_units, EN_o_units):
        super(DAGMM, self).__init__()
        with self.init_scope():

            # Compression Network
            self.ec_l1 = L.Linear(None,CN_h_units)
            self.ec_l2 = L.Linear(None,CN_z_units)

            self.dc_l1 = L.Linear(None,CN_h_units)
            self.dc_l2 = L.Linear(None,CN_o_units)

            # Estimation Network
            self.en_l1 = L.Linear(None,EN_h_units)
            self.en_l2 = L.Linear(None,2)

    dirGMMparameters = "./tmp_GMMparameters/"


    def lossFunc(self,gpu=-1):
        if gpu >= 0:
            xp = cuda.cupy
        else:
            xp = np

        def lf(x):
            y, energy, sigma = self.fwd(x, isTraining=True, gpu=gpu)
            # 		reconstLoss = F.mean_squared_error(x, y) #損失関数
            reconstLoss = F.sum(F.squared_error(x,y)) / len(x) # 再構築誤差(ミニバッチ平均)
            avgEnergy = F.sum(energy) / len(x) # エネルギー(ミニバッチ平均)
            # 正則化項について
            NumOfClass, zDim, _ = sigma.shape
            diagMat = Variable(xp.array(np.array(list(np.eye(zDim))*NumOfClass).reshape(NumOfClass,zDim,zDim).astype(np.float32)))
            reg = F.sum((1/sigma) * diagMat)
            #重みパラメータ
            lambda_1 =  0.1
            lambda_2 =  0.005

            self.loss = reconstLoss + lambda_1 * avgEnergy + lambda_2 * reg

            chainer.report( {'loss': self.loss}, observer=self )
            return self.loss
        return lf



    def fwd(self, x, isTraining=False, gpu=-1):
        if gpu >= 0:
            xp = cuda.cupy
        else:
            xp = np


        # Compression Network
        # エンコード
        zc = self.Encoder(x)
        # デコード
        y = self.Decoder(zc)

        # 再構築誤差を計算 relativeユークリッド距離とコサイン類似度
        rEucDist = self.relativeEuclideanDistance(x,y)
        CosSim = self.cosineSimilarity(x,y)
        #潜在変数と再構築誤差を合体
        z = F.concat((zc,rEucDist,CosSim), axis=1)


        # Estimation Network
        gamma = self.Estimation(z)


        NumOfData, NumOfClass = gamma.shape
        _, zDim = z.shape


        # GMM
        # 各サンプルの各分布への帰属確率から、各分布の混合比、平均ベクトル、共分散行列を得る
        if chainer.config.train:
            phi = self._phi(gamma)
            mu = self._mu(z,gamma)
            sigma = self._sigma(z,gamma,mu)

            os.makedirs(self.dirGMMparameters, exist_ok=True)
            if chainer.cuda.available:
                np.savetxt(self.dirGMMparameters + "phi.csv", np.array((phi.data).get()), delimiter=",")
                np.savetxt(self.dirGMMparameters + "mu.csv", np.array((mu.data).get()), delimiter=",")
                np.savetxt(self.dirGMMparameters + "sigma.csv", np.array((sigma.data).get()).reshape(NumOfClass,-1), delimiter=",")
            else:
                np.savetxt(self.dirGMMparameters + "phi.csv", np.array(phi.data), delimiter=",")
                np.savetxt(self.dirGMMparameters + "mu.csv", np.array(mu.data), delimiter=",")
                np.savetxt(self.dirGMMparameters + "sigma.csv", np.array(sigma.data).reshape(NumOfClass,-1), delimiter=",")

        else:
            phi = Variable(xp.array(np.loadtxt(self.dirGMMparameters + "phi.csv",delimiter=",").astype(np.float32)))
            mu = Variable(xp.array(np.loadtxt(self.dirGMMparameters + "mu.csv",delimiter=",").astype(np.float32)))
            sigma = Variable(xp.array((np.loadtxt(self.dirGMMparameters + "sigma.csv",delimiter=",")).reshape(NumOfClass,zDim,zDim).astype(np.float32)))




        # エネルギーを計算
        # eps = 1e-3 #ランク落ちまたは、行列式が0になってしまう対策
        # sigma = sigma + Variable(xp.array(np.array(list(np.eye(zDim))*NumOfClass).reshape(NumOfClass,zDim,zDim).astype(np.float32)) * eps)
        sigmaInv = F.batch_inv(sigma) # shape(3D) -> NumOfClass, zDim, zDim
        z_broadcast = F.broadcast_to(z, (NumOfClass, NumOfData, zDim)) # shape(3D) -> NumOfClass, NumOfData, zDim
        mu_broadcast = F.transpose(F.broadcast_to(mu, (NumOfData, NumOfClass, zDim)),axes=(1,0,2)) # shape(3D) -> NumOfClass,
        sa = z_broadcast - mu_broadcast
        listForEnr1 = [ F.matmul(sa[i],sigmaInv[i]) for i in range(NumOfClass)] # shape(3D) -> NumOfClass, NumOfData, zDim
        listForEnr2 = [ F.sum(listForEnr1[i]*sa[i],axis=1) for i in range(NumOfClass)] # shape(2D) -> NumOfClass, NumOfData
        varForEnr = F.stack([listForEnr2[i] for i in range(len(listForEnr2))], axis=0) # リストからVariableへ変換 # shape(2D) -> NumOfClass, NumOfData
        numer = F.exp(-(1/2) * varForEnr) # 分子の計算  # shape(2D) -> NumOfClass, NumOfData
        denom = F.transpose(F.broadcast_to(F.sqrt(F.batch_det(2*math.pi*sigma)), (NumOfData,NumOfClass)))# 分母の計算  # shape(2D) -> NumOfClass, NumOfData
        phi_broadcast = F.transpose(F.broadcast_to(phi,(NumOfData,NumOfClass))) # shape(2D) -> NumOfClass, NumOfData
        energy = -1 * F.log(F.sum(phi_broadcast * (numer / denom), axis=0, keepdims=True)) # shape(2D) -> 1, NumOfData
        energy = F.transpose(energy) # shape(2D) -> NumOfData,1

        if isTraining:
            return y, energy, sigma
        else:
            return z, energy, gamma, y





    def Encoder(self,x):
        h = F.tanh(self.ec_l1(x))
        z = self.ec_l2(h)
        return z


    def Decoder(self,z):
        h = F.tanh(self.dc_l1(z))
        y = self.dc_l2(h)
        return y

    def relativeEuclideanDistance(self, x, y):
        L2NormXY = F.sqrt(F.sum(F.squared_error(x,y), axis=1, keepdims=True))
        L2NormX = F.reshape(F.sqrt(F.batch_l2_norm_squared(x)), (-1,1))
        return L2NormXY / (L2NormX)

    def cosineSimilarity(self, x, y):
        innerProduct = F.reshape(F.sum(x*y, axis=1), (-1,1))
        L2NormX = F.reshape(F.sqrt(F.batch_l2_norm_squared(x)), (-1,1))
        L2NormY = F.reshape(F.sqrt(F.batch_l2_norm_squared(y)), (-1,1))
        return innerProduct / (L2NormX * L2NormY)

    def Estimation(self,z):
        h = F.dropout(F.tanh(self.en_l1(z)), 0.5)
        gamma = F.softmax(self.en_l2(h))
        return gamma


    def _phi(self,gamma):
        # 各分布の混合比phi (スカラー/各クラス)
        phi = F.average(gamma, axis=0)
        return phi

    def _mu(self,z,gamma):
        # 平均ベクトルmu (ベクトル/各クラス)
        NumOfData, NumOfClass = gamma.shape
        _, zDim = z.shape
        gamma_T = F.transpose(gamma) # shape(2D) -> NumOfClass, NumOfData
        gamma_broadcast = F.transpose(F.broadcast_to(gamma_T, (zDim, NumOfClass, NumOfData)), axes=(1,2,0)) # shape(3D) -> NumOfClass, NumOfData, zDim
        z_broadcast = F.broadcast_to(z, (NumOfClass, NumOfData, zDim)) # shape(3D) -> NumOfClass, NumOfData, zDim
        gamma_sum_broadcast = F.transpose(F.broadcast_to(F.sum(gamma, axis=0),  (zDim,NumOfClass))) # shape(2D) -> NumOfClass, zDim
        mu = F.sum(gamma_broadcast * z_broadcast, axis=1) / gamma_sum_broadcast # shape(2D) -> NumOfClass, zDim
        return mu

    def _sigma(self,z,gamma,mu):
        # 共分散行列sigma (行列/各クラス)
        NumOfData, NumOfClass = gamma.shape
        _, zDim = z.shape
        gamma_T = F.transpose(gamma) # shape(2D) -> NumOfClass, NumOfData
        z_broadcast = F.broadcast_to(z, (NumOfClass, NumOfData, zDim)) # shape(3D) -> NumOfClass, NumOfData, zDim
        mu_broadcast = F.transpose(F.broadcast_to(mu, (NumOfData, NumOfClass, zDim)),axes=(1,0,2)) # shape(3D) -> NumOfClass, NumOfData, zDim
        sa = z_broadcast - mu_broadcast # shape(3D) -> NumOfClass, NumOfData, zDim
        listForCov1 = [[F.matmul(F.reshape(j,(-1,1)),F.reshape(j,(1,-1)))  for j in i] for i in sa] # 外積の計算
        listForCov2 = [F.stack([listForCov1[i][j] for j in range(len(listForCov1[i]))], axis=0) for i in range(len(listForCov1))] # リストからVariableへ変換
        vecProduct = F.stack([listForCov2[i] for i in range(len(listForCov2))], axis=0) # リストからVariableへ変換 # shape(4D) -> NumOfClass, NumOfData, zDim, zDim
        gamma_broadcastForCov = F.transpose(F.broadcast_to(gamma_T, (zDim, zDim, NumOfClass, NumOfData)), axes=(2,3,1,0))
        gamma_sum_broadcastForCov = F.transpose(F.broadcast_to(F.sum(gamma, axis=0),  (zDim,zDim,NumOfClass)))
        sigma = F.sum(vecProduct*gamma_broadcastForCov, axis=1) / gamma_sum_broadcastForCov # shape(3D) -> NumOfClass, zDim, zDim
        return sigma
