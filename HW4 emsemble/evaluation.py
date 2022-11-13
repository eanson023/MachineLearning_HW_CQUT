#! python3
# -*- encoding: utf-8 -*-
"""
Created on Sat Oct  29 09:20:46 2022

@author: eanson
"""
# 评估准确率
def evaluate(model, X, y):
    y_predict = model.predict(X)
    accuracy = (y_predict == y).sum()/len(y)
    return accuracy