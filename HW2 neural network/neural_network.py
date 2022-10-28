#! python3
# -*- encoding: utf-8 -*-
"""
Created on Mon Oct  10 19:16:53 2022

HW2 MNIST手写体数字识别

@author: eanson
"""
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import numpy as np

###########################################################################################
# hyperparamaters
batch_size = 64
epochs = 3
learning_rate = 0.02
###########################################################################################

# 输入数据集
train_datasets = datasets.MNIST(root='HW2 neural network/data',
                                train=True, transform=transforms.ToTensor(), download=False)
test_datasets = datasets.MNIST(root='HW2 neural network/data',
                               train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(
    dataset=train_datasets, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_datasets, batch_size=batch_size, shuffle=False)


###########################################################################################
# 数据集可视化
images, labels = iter(train_loader).next()
for i in range(len(labels)):
    plt.subplot(10, 10, i+1)
    plt.imshow(images[i, ...].reshape((28, 28)), cmap="gray")
    # 关闭坐标轴
    plt.axis('off')
plt.show()
###########################################################################################
# 对数似然函数
loss_function = nn.NLLLoss()


# 定义深层次可分离卷积
class DeepWise_PointWise_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepWise_PointWise_Conv, self).__init__()
        self.deepwise_layer = nn.Conv2d(
            in_channels,
            out_channels=in_channels,
            kernel_size=3,
            # 用分组计算性质来实现输入通道和输出通道相同
            groups=in_channels,
        )
        self.pointwise_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, X):
        return self.pointwise_layer(self.deepwise_layer(X))


# 模型建立
class YanNet_is_not_all_you_need(nn.Module):
    def __init__(self):
        super(YanNet_is_not_all_you_need, self).__init__()
        # 输入(1,28,28) 输出 (32, 28, 28)
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU()
        )
        # 输入(32, 28, 28) 输出(64, 13 ,13)
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 输入(64,13,13) 输出(8,11,11)
        self.layer3 = nn.Sequential(
            # 使用深层次可分离卷积
            DeepWise_PointWise_Conv(64, 8),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            # 平铺
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=8*11*11, out_features=10),
            nn.LogSoftmax(dim=1)
            )

    # 前向传播函数
    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        output = self.layer4(X)
        return output


def evaluate(model, test_loader, log, epoch):
    right_num = 0
    for batch_idx, (X, y) in enumerate(test_loader):
        # Sets the module in evaluation mode.
        model.eval()
        output = model(X)
        # 输出的下标即预测值
        y_pred = torch.max(output, dim=1)[1]
        right_num = right_num + torch.eq(y_pred, y).sum().item()
        if batch_idx % 10 == 0:
            # 计算在测试集上的loss
            loss = loss_function(output, y)
            log.append(
                [((epoch*len(test_loader.dataset))+(batch_size*batch_idx+1)), loss.detach().numpy()])
    print("test accurecny:[{:0.6f}]%".format(
        right_num*100/len(test_loader.dataset)))

###########################################################################################
# 模型训练


def train():
    # 模型
    model = YanNet_is_not_all_you_need()
    print(model)
    # 随机梯度下降优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    log_train, log_test = [], []
    for epoch in range(epochs):
        for batch_idx, (X, y) in enumerate(train_loader):
            # Sets the module in train mode.
            model.train()
            # 清空梯度
            optimizer.zero_grad()
            # 调用模型输出预测值
            output = model(X)
            # 计算loss
            loss = loss_function(output, y)
            # 反向传播计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            if batch_idx % 10 == 0:
                log_train.append(
                    [((epoch*len(train_loader.dataset))+(batch_size*batch_idx+1)), loss.detach().numpy()])
            if batch_idx % 100 == 0:
                print("epoch[{}]: batch:[{}/{}] loss:[{:0.10f}]".format(epoch +
                      1, batch_size*batch_idx+1, len(train_loader.dataset), loss))
        evaluate(model, test_loader, log_test, epoch)
    # 画下loss
    log_train = np.array(log_train)
    # log_test = np.array(log_test)
    plt.title(
        "loss changes in [{}] epochs usiing cross-entropy loss]".format(epochs))
    plt.plot(log_train[:, 0], log_train[:, 1])
    # plt.plot(log_test[:, 0], log_test[:, 1])
    # plt.legend(['loss_train', 'loss_test'])
    plt.ylabel("loss")
    plt.xlabel("ierations")
    plt.show()


if __name__ == "__main__":
    train()
