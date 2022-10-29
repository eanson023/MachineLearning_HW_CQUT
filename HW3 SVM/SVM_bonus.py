#! python3
# -*- encoding: utf-8 -*-
"""
Created on Fri Oct  28 21:07:29 2022

HW3 支持向量机作业 pt.3

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
file_name = 'data3.mat'
shuffle_num = 50
kernel = 'rbf'
C_max = 1000
######################################################################################


def main():
    log = []
    sigmas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]
    for C in range(1, C_max+1):
        for sigma in sigmas:
            ratios = []
            for _ in range(shuffle_num):
                # 随机加载shuffle_num次数据
                X_train, y_train, X_cv, y_cv = data.load_data(
                    file_name, train_proportion=0.7, visualize=False)
                # 定义分类器
                clf = SVM.SVC(C=C, kernel=kernel, gamma=sigma)
                # 训练模型
                clf.fit(X_train, y_train)
                ratios.append([evaluate(clf, X_train, y_train),
                               evaluate(clf, X_cv, y_cv)])
            ratios = np.array(ratios)
            # 求出平均准确率
            means = np.mean(ratios, axis=0) * 100
            log.append([C, sigma, means[0], means[1]])
            print("paramater C=[{:4d}]\t sigma=[{}]\tmean training accuracy:{:.6f}%\tmean test accuracy:{:.6f}%".format(
                C, sigma, means[0], means[1]))
    log = np.array(log)
    X_train, y_train, X_cv, y_cv = data.load_data(
        file_name, train_proportion=0.7, visualize=False)
    # 查找验证集上准确率最高的参数C
    the_best = log[np.argmax(log, axis=0)[3], :]
    C_best, sigma_best = the_best[0], the_best[1]
    clf = SVM.SVC(C=C_best, kernel=kernel)
    clf.fit(X_train, y_train)
    print("the best paramater C is:{}\tsigma=[{}]\ttraining accuracy:{:.6f}%\ttest accuracy:{:.6f}%".format(
        C_best, sigma_best, the_best[2], the_best[3]))
    # 可视化在最优sigma的情况下的数据集准确率变化
    indexs = np.where(log[:, 1] == sigma_best)[0]
    plt.plot(log[indexs, 0], log[indexs, 2])
    plt.plot(log[indexs, 0], log[indexs, 3])
    plt.legend(['training accuracy', 'test accuracy'])
    plt.xlabel("C")
    plt.ylabel("ratio")
    plt.show()
    viz.visualize_boundary(clf, X_train, y_train, kernel)
    viz.visualize_boundary(clf, X_cv, y_cv, kernel, 'test')


if __name__ == "__main__":
    main()
