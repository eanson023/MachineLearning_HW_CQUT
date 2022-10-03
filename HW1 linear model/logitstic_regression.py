#! python3
# -*- encoding: utf-8 -*-
"""
Created on Mon Oct  03 11:31:30 2022

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
data_file = os.path.join('HW1 linear model', 'data2.txt')
data = pd.read_csv(data_file,
                   header=None,
                   names=["Score1", "Score2", "Price"])
# 训练集按照比例划分
m = math.floor(len(data)*train_proportion)

############################################################################
# 读取数据集
X_train_pd, y_train_pd, X_cv_pd, y_cv_pd = data.iloc[:m,
                                                     :2], data.iloc[:m, -1], data.iloc[m+1:, :2], data.iloc[m+1:, -1]

# 转换为张量格式 引入特征 x0=1 （方便之后向量化）
X_train, y_train = torch.cat(
    [torch.ones(m, 1), torch.tensor(X_train_pd.values, dtype=torch.float64)], dim=1), torch.tensor(y_train_pd.values, dtype=torch.float64).reshape(-1, 1)
X_cv, y_cv = torch.cat([torch.ones(len(X_cv_pd), 1), torch.tensor(
    X_cv_pd.values, dtype=torch.float64)], dim=1), torch.tensor(y_cv_pd.values, dtype=torch.float64).reshape(-1, 1)

############################################################################
# 特征缩放
X_train[:, 1:] = (X_train[:, 1:]-X_train[:, 1:].mean())/X_train[:, 1:].std()
X_cv[:, 1:] = (X_cv[:, 1:]-X_cv[:, 1:].mean())/X_cv[:, 1:].std()

############################################################################
# 可视化


def visualize(X, y, w, title_suffix):
    plt.title("prediction "+title_suffix)
    plt.xlabel("score1")
    plt.ylabel("score2")
    x_points = X[:, 1].numpy()
    y_points = X[:, 2].numpy()
    plt.scatter(x_points, y_points, c=y.flatten())
    x1 = np.arange(np.min(x_points), np.max(x_points), 0.1)
    # h_theta=g(theta0*x1+theta1*x2+theta3*x3)移项而来 当然x1=1
    w = w.clone().detach().numpy()
    x2 = -(w[0]*1+w[1]*x1)/w[2]
    plt.plot(x1, x2)
    plt.show()


############################################################################
# 梯度下降法

# 迭代次数
iterations = 10000
# 学习率
lr = 0.01
# lambda
lambda_ = 1
# 初始化theta
# size: feature_size+1 x 1
w = torch.randn((X_train.size(1), 1), dtype=torch.float64)


def loss_function(y_pred, y):
    """
    $$
    J(\theta )=-\frac{1}{m}\sum_{i=1}^{m}[y{}^{(i)}\log_{}({h_{\theta } }(x{}^{(i)}))+(1-y{}^{(i)})\log_{}(1-{h_{\theta } }(x{}^{(i)}))]  
    $$
    """
    return -torch.mean(y*torch.log(y_pred)+(1-y)*torch.log(1-y_pred))


# 梯度下降
def gradient_descend(lr, batch_size):
    global w
    with torch.no_grad():
        w -= lr*w.grad/batch_size
        w.grad.zero_()


# 假设函数(加上sigmoid)
def hypothesis(X):
    return 1.0/(1.0+torch.exp(-torch.mm(X, w)))


def predict(y):
    return (y >= 0.5).type(torch.float64)


# 训练
def train(net, loss, X_train, y_train, X_cv, y_cv, num_epochs):
    global w
    log = []
    for epoch in range(num_epochs):
        y_pred = net(X_train)
        # 计算梯度
        grad = torch.mean((y_pred-y_train)*X_train, dim=0).reshape(-1, 1)
        # 梯度下降
        w = w-lr*grad
        l = loss_function(y_pred, y_train)

        y_pred_cv = net(X_cv)
        l_cv = loss_function(y_pred_cv, y_cv)
        print(f'epoch:{epoch+1},train loss:{l}\t cross validation loss:{l_cv}')
        log.append([epoch+1, l.detach().numpy(), l_cv.detach().numpy()])
    # 画下loss
    log = np.array(log)
    plt.plot(log[:, 0], log[:, 1])
    plt.plot(log[:, 0], log[:, 2])
    plt.legend(['loss_train','loss_cv'])
    plt.show()


# train(hypothesis, loss_function, X_train, y_train, X_cv, y_cv, iterations)
# visualize(X_cv, y_cv, w, 'on cross validation data')

############################################################################
# 梯度下降法(L2正则化)


# 训练
def train2(net, loss, X_train, y_train, X_cv, y_cv, num_epochs):
    global w
    log = []
    for epoch in range(num_epochs):
        y_pred = net(X_train)
        # 计算梯度 加上正则化
        grad = torch.mean((y_pred-y_train)*X_train,
                          dim=0).reshape(-1, 1)+2*lambda_*w
        # 排除 theta0
        grad[0] = grad[0]-2*lambda_*w[0]
        # 梯度下降
        w = w-lr*grad
        # 加上正则化
        l = loss_function(y_pred, y_train) + lambda_*torch.sum(w**2)
        # 排除 theta0
        l = l - w[0]**2
        y_pred_cv = net(X_cv)
        l_cv = loss_function(y_pred_cv, y_cv)+ lambda_*torch.sum(w**2)
        l_cv = l_cv - w[0]**2
        print(f'epoch:{epoch+1},train loss:{l}\t cross validation loss:{l_cv}')
        log.append([epoch+1, l.detach().numpy(), l_cv.detach().numpy()])
    # 画下loss
    log = np.array(log)
    plt.plot(log[:, 0], log[:, 1])
    plt.plot(log[:, 0], log[:, 2])
    plt.legend(['loss_train','loss_cv'])
    plt.show()


train2(hypothesis, loss_function, X_train, y_train, X_cv, y_cv, iterations)
# visualize(X_train, y_train, w, 'on training data')
visualize(X_cv, y_cv, w, 'on cross validation data')
