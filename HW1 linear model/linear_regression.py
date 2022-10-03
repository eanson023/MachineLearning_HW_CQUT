#! python3
# -*- encoding: utf-8 -*-
"""
Created on Mon Sep  26 08:02:10 2022

@author: eanson
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import math

import os

# 训练集比例
train_proportion = 0.7

############################################################################

# 读取数据集
data_file = os.path.join('HW1 linear model', 'data1.txt')
data = pd.read_csv(data_file,
                   header=None,
                   names=["Room Size", "Room Amount", "Price"])
# 训练集按照比例划分
m = math.floor(len(data)*train_proportion)


############################################################################
# feature scaling 特征缩放
data_ = torch.tensor(data.values, dtype=torch.float32)
# 每列的均值,每列的标准差
data_mean, data_std = data_.mean(dim=0), data_.std(dim=0)
"""
$$
x_{n} = \frac{x_{n}-u_{n}}{s_{n}} （其中u_{n}是平均值，s_{n}是标准差）
$$
"""
data_norm = (data_-data_mean)/data_std

# 存储房间大小和房价的平均值和标准差用于之后画图还原坐标轴
room_size_mean, price_mean = data_mean[0], data_mean[2]
room_size_std, price_std = data_std[0], data_std[2]

############################################################################
# 读取数据集

# 转换为张量格式 引入特征 x0=1 （方便之后向量化）
X_train, y_train = torch.cat(
    [torch.ones(m, 1), data_norm[:m, 0:2]], dim=1), data_norm[:m, 2].reshape(-1, 1)
X_cv, y_cv = torch.cat([torch.ones(
    data_.shape[0]-(m+1), 1), data_norm[m+1:, 0:2]], dim=1), data_norm[m+1:, 2].reshape(-1, 1)

############################################################################
# 可视化


def visualize(X, y, w, title_suffix):
    # 计算验证集预测值
    y_pred = torch.mm(X, w)
    x_points = (X[:, 1]*room_size_std+room_size_mean).numpy()
    y_points = (y*price_std+price_mean).detach().numpy()
    y_pred_points = (y_pred*price_std+price_mean).detach().numpy()

    plt.title("house price prediction "+title_suffix)
    plt.xlabel("room size(m²)")
    plt.ylabel("house price($)")
    o1, = plt.plot(x_points, y_points, 'o', c='b')
    o2, = plt.plot(x_points, y_pred_points, 'o', c='r')
    plt.legend(handles=[o1, o2], labels=[
               'label', 'precision'], loc='upper left')
    plt.show()


############################################################################
# 正规方程法
"""
$$
w=(X^TX)^{-1}X^Ty
$$
"""
w = torch.mm(X_train.T, X_train).inverse().mm(
    X_train.T).mm(y_train)

visualize(X_cv, y_cv, w, 'on cross validation data with normal equation')


############################################################################
# 梯度下降法

# 迭代次数
iterations = 10000
# 学习率
lr = 0.01
# 初始化theta
# size: feature_size+1 x 1
w = torch.rand((X_train.size(1), 1), requires_grad=True)


def loss_function(y_pred, y):
    m = len(y_pred)
    return 1/(2*m)*torch.sum((y_pred-y)**2)


# 梯度下降
def gradient_descend(lr, batch_size):
    global w
    with torch.no_grad():
        w -= lr*w.grad/batch_size
        w.grad.zero_()


# 模型
def model(X):
    return torch.mm(X, w)


# 训练
def train(net, loss, updater, X_train, y_train, X_cv, y_cv, num_epochs):
    global w
    log = []
    for epoch in range(num_epochs):
        y_pred = net(X_train)
        # 向量化
        grad = lr*torch.mean((y_pred-y_train)*X_train, dim=0).reshape(-1, 1)
        # 梯度下降
        w = w - grad
        l = loss_function(y_pred, y_train)
        print(f'epoch:{epoch+1},loss:{l}')
        log.append([epoch+1, l.detach().numpy()])
    # 画下loss
    log = np.array(log)
    plt.plot(log[:, 0], log[:, 1])
    plt.show()


train(model, loss_function, gradient_descend,
      X_train, y_train, X_cv, y_cv, iterations)
visualize(X_cv, y_cv, w, 'on cross validation data with gradient descent')

# log = []
# for i in range(iterations):
#     # 假设函数
#     hypothesis = torch.mm(X_train, w)
#     # 向量化
#     grad = lr*torch.mean((hypothesis-y_train)*X_train, dim=0).reshape(-1, 1)
#     # 梯度下降
#     w = w - grad
#     loss = loss_function(hypothesis, y_train)
#     print(f'iter:{i+1},loss:{loss}')
#     log.append([i+1, loss])

# # 画下loss
# log = np.array(log)
# plt.figure(figure_idx)
# figure_idx += 1
# plt.plot(log[:, 0], log[:, 1])
# plt.show()

# visualize(X_cv, y_cv, w, 'on cross validation data with gradient descent')
