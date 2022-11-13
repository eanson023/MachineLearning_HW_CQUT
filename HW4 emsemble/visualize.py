#! python3
# -*- encoding: utf-8 -*-
"""
Created on Sat Oct  29 09:16:32 2022

@author: eanson
"""
import numpy as np
from matplotlib import pyplot as plt
from evaluation import evaluate


def visualize_boundary(model, X, y, classifier, training=True):
    # +0.1 -0.1的目的：避免将样本绘制到图的边缘区域
    x1 = np.linspace(np.min(X[:, 0])-0.1, np.max(
        X[:, 0])+0.1, 100).reshape(-1, 1)  # 100 x 1
    x2 = np.linspace(np.min(X[:, 1])-0.1, np.max(
        X[:, 1])+0.1, 100).reshape(-1, 1)  # 100 x 1
    x1, x2 = np.meshgrid(x1, x2)
    # 列堆叠成坐标网格
    X_coordinate = np.column_stack((x1.flatten(), x2.flatten()))
    # 使用SVM直接输出预测值
    y_predict = model.predict(X_coordinate).reshape(x1.shape)
    # 绘制图像
    plt.contourf(x1, x2, y_predict, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20)
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.title("using random forest classifier on {} dataset, accurency:[{:.2f}%]".format(
        'training' if training else 'test', evaluate(model, X, y)*100))
    plt.show()


def visualize_iris(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()
