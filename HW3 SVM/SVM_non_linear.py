#! python3
# -*- encoding: utf-8 -*-
"""
Created on Fri Oct  28 20:39:40 2022

HW3 支持向量机作业 pt.2

@author: eanson
"""
import numpy as np
from sklearn import svm as SVM
from matplotlib import pyplot as plt

from evaluation import evaluate
import visualize as viz
import data

######################################################################################
# 加载数据集
file_name = 'data2.mat'
X_train, y_train, X_cv, y_cv = data.load_data(
    file_name, train_proportion=1, visualize=False)


######################################################################################


######################################################################################
def main():
    C = 1000.0
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for kernel in kernels:
        kernel = 'rbf'
        # 定义分类器
        clf = SVM.SVC(C=C, kernel=kernel, gamma=100000.0)
        # 训练模型
        clf.fit(X_train, y_train)
        viz.visualize_boundary(clf, X_train, y_train, kernel)
        # ratio = evaluate(clf, X_cv, y_cv)
        # print("the accuracy of test dataset with {:10s} kernel:\t{:.6f}%".format(
        #     kernel, ratio*100))
        # viz.visualize_boundary(clf, X_cv, y_cv, kernel, 'test')


if __name__ == "__main__":
    main()
