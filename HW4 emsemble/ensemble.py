#! python3
# -*- encoding: utf-8 -*-
"""
Created on Fri Nov  11 08:47:39 2022

HW4 集成学习

@author: eanson
"""

import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

###########################导入不同基分类器开始###############################
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Perceptron
###########################导入不同基分类器结束###############################

import math
from os import path
import data
import visualize as viz

# import inspect
# from sklearn.utils import all_estimators
# for name, clf in all_estimators(type_filter='classifier'):
#     if name not in ['ClassifierChain', 'MultiOutputClassifier', 'OneVsOneClassifier', 'OneVsRestClassifier', 'OutputCodeClassifier', 'StackingClassifier'] and 'sample_weight' in inspect.getargspec(clf().fit)[0]:
#         print(name, clf)

train_proportion = 0.7

##########################################################################
# 输入数据集 （10）

######################################################################################
# 加载数据集
file_name = 'data1.mat'
X_train, y_train, X_cv, y_cv = data.load_data(
    file_name, train_proportion=0.7, visualize=False)
##########################################################################


def train():
    # 定义分类器
    clf = RandomForestClassifier()
    # 训练模型
    clf.fit(X_train, y_train)
    viz.visualize_boundary(clf, X_cv, y_cv, '', False)


def grid_search_cv():
    from sklearn.model_selection import GridSearchCV
    param_test1 = {
        'n_estimators': range(1, 100, 2),
    }
    param_test2 = {
        #
        'criterion': ["gini", "entropy"],
        # 决策树最大深度
        'max_depth': range(1, 20, 2),
        # 叶子节点最少样本数
        'min_samples_leaf': range(1, 10)
    }
    # oob_score：否采用袋外样本来评估模型的好坏
    clf = RandomForestClassifier(oob_score=False, random_state=10)
    gsearch = GridSearchCV(estimator=clf,
                           param_grid=param_test1, scoring='roc_auc', cv=5)
    print('grid search beginning')
    gsearch.fit(X_train, y_train)
    print('-'*100)
    best_n_estimators = gsearch.best_params_['n_estimators']
    print('the best n_estimators is:{} scores:{}'.format(
        best_n_estimators, gsearch.best_score_))
    print('-'*100)
    print('using it to fit the other best params...')
    clf = RandomForestClassifier(
        n_estimators=best_n_estimators, oob_score=False, random_state=10)
    gsearch = GridSearchCV(estimator=clf,
                           param_grid=param_test2, scoring='roc_auc', cv=5)
    gsearch.fit(X_train, y_train)
    print('-'*100)
    best_criterion = gsearch.best_params_['criterion']
    best_max_depth = gsearch.best_params_['max_depth']
    best_min_samples_leaf = gsearch.best_params_['min_samples_leaf']
    print('the best params is:{} scores:{}'.format(
        gsearch.best_params_, gsearch.best_score_))
    print('-'*100)
    clf = RandomForestClassifier(
        n_estimators=best_n_estimators,
        criterion=best_criterion,
        max_depth=best_max_depth,
        min_samples_leaf=best_min_samples_leaf,
        oob_score=False,
        random_state=10)
    clf.fit(X_train, y_train)
    score = clf.score(X_cv, y_cv)
    print('using best params above,the score got on test set is {}%'.format(score*100))
    print('-'*100)


def bonus():
    scores1, scores2 = [], []
    for _ in range(100):
        X_train, y_train, X_cv, y_cv = data.load_iris_data(visualize=False)
        # 使用默认参数
        clf = AdaBoostClassifier()
        clf.fit(X_train, y_train)
        score = clf.score(X_cv, y_cv)
        scores1.append(score)
        # 使用默认参数
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        score = clf.score(X_cv, y_cv)
        scores2.append(score)
    print('using adaboost classifier, the score got on test set is {}%'.format(
        np.asarray(scores1).mean()*100))
    print('using random forest classifier, the score got on test set is {}%'.format(
        np.asarray(scores2).mean()*100))


if __name__ == "__main__":
    # grid_search_cv()
    bonus()
