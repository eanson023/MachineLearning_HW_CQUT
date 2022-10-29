#! python3
# -*- encoding: utf-8 -*-
"""
Created on Fri Oct  28 16:39:59 2022

HW3 支持向量机作业 pt.1

@author: eanson
"""
import numpy as np
from sklearn import svm as SVM

import visualize as viz
from evaluation import evaluate
import data

######################################################################################
# 加载数据集
file_name = 'data1.mat'
X_train, y_train, X_cv, y_cv = data.load_data(
    file_name, train_proportion=1, visualize=False)


######################################################################################
def main():
    # 超参数
    C = 100.0
    # 定义分类器
    clf = SVM.LinearSVC(C=C)
    # 训练模型
    clf.fit(X_train, y_train)
    viz.visualize_boundary(clf, X_train, y_train, 'linear')
    # ratio = evaluate(clf, X_cv, y_cv)
    # print("test accuracy:\t{:.6f}%".format(ratio*100))
    # viz.visualize_boundary(clf, X_cv, y_cv, 'linear', 'test')


if __name__ == "__main__":
    main()
