#coding:utf-8
#!/usr/bin/env python

import numpy as np


def main():

    # csvデータ読み込み
    data = np.loadtxt('./data_clean_imputed.csv',delimiter=',')
    NumOfData, NumOfAttribute = data.shape

    # 説明変数と目的変数に分割
    ExplanatoryVariables, CriterionVariables = data[:,:NumOfAttribute-1], data[:,-1]

    # 下記の属性を取り除く
    # 1 Age: Age in years , linear
    # 2 Sex: Sex (0 = male; 1 = female) , nominal
    # 3 Height: Height in centimeters , linear
    # 4 Weight: Weight in kilograms , linear
    ExplanatoryVariables = ExplanatoryVariables[:,4:]

    # csvファイル書き出し
    np.savetxt('./../ExplanatoryVariables.csv',ExplanatoryVariables,delimiter=',')
    np.savetxt('./../CriterionVariables.csv',CriterionVariables,delimiter=',')


if __name__ == "__main__":
    main()
