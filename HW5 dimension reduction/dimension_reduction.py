#! python3
# -*- encoding: utf-8 -*-
"""
Created on Wed Nov  23 21:25:25 2022

手搓PCA

@author: eanson
"""

import numpy as np
import scipy.io as sio
from os import path
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def load_data(mat_name):
    data_dict = sio.loadmat(path.join('HW5 dimension reduction', mat_name))
    X = np.array(data_dict['X'])
    return X


def pca(X, k):
    """
    k:低维空间维度
    """
    # 复制一下
    X = X.copy()
    # 1.对所有样本去中心化
    mean = np.mean(X, axis=0)
    X -= mean
    # 2.计算样本的协方差矩阵
    # rowvar=0代表每行为样本
    sigma = np.cov(X, rowvar=0)
    # 3.对协方差矩阵做特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(sigma)
    # 4.取出最大的k个特征值所对应的特征向量w1,w2,...,wk
    # 逆序
    inds = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, inds]
    # 降维后的数据
    X_reduct = np.dot(X, eigenvectors[:, :k])
    # 重构数据
    X_approx = np.dot(X_reduct, eigenvectors[:, :k].T)+mean
    return X_reduct, X_approx


def visualize(X, k, title):
    """
    k:低维空间维度
    """
    # 复制一下
    X_origin = X.copy()
    _, X_recovery = pca(X, k)
    plt.scatter(X_origin[:, 0], X_origin[:, 1], c='b')
    plt.plot(X_recovery[:, 0], X_recovery[:, 1], c='r')
    plt.title(title)
    plt.show()


def choose_best_k(X):
    best_k = 1
    best_ratio = 1 << 31
    variance = np.sum(np.linalg.norm(X, ord=2, axis=1))
    for k in range(1, X.shape[1]):
        _, X_approx = pca(X, k)
        ratio = np.sum(np.linalg.norm(X-X_approx, ord=2, axis=1)) / variance
        if best_ratio > ratio:
            best_ratio = ratio
            best_k = k
        print('k={},ratio={:.2f}%'.format(k, ratio*100))
    return best_k, best_ratio


def sklearn_mat1():
    file_name = 'data1.mat'
    X = load_data(file_name)
    pca = PCA(n_components=1)
    # 降维
    X_reduct = pca.fit_transform(X)
    # 还原
    X_approx = np.dot(X_reduct, pca.components_)+np.mean(X, axis=0)
    plt.scatter(X[:, 0], X[:, 1], c='b')
    plt.plot(X_approx[:, 0], X_approx[:, 1], c='r')
    plt.show()


def main():
    sklearn_mat1()


if __name__ == "__main__":
    main()
