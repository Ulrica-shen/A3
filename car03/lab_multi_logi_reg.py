import inspect
import unittest

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,multilabel_confusion_matrix
import time
from car03.multi_logistic_reg import LogisticRegression


def prep_report_data():
    # Step 1: Prepare data
    # import some data to play with
    iris = datasets.load_iris()
    # 矩阵切片
    X = iris.data[:, 2:]  # we only take the first two features.
    y = iris.target  # now our y is three classes thus require multinomial

    # Split data into training and test datasets
    idx = np.arange(0, len(X), 1)
    np.random.shuffle(idx)
    idx_train = idx[0:int(.7 * len(X))]
    idx_test = idx[len(idx_train):len(idx)]

    X_train = X[idx_train]
    X_test = X[idx_test]
    y_train = y[idx_train]  # 只有0 1 2三种类型
    y_test = y[idx_test]

    # feature scaling helps improve reach convergence faster
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # add intercept to our X
    intercept = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((intercept, X_train), axis=1)  # add intercept
    intercept = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((intercept, X_test), axis=1)  # add intercept

    # make sure our y is in the shape of (m, k)
    # we will convert our output vector in
    # matrix where no. of columns is equal to the no. of classes.
    # The values in the matrix will be 0 or 1. For instance the rows
    # where we have output 2 the column 2 will contain 1 and the rest are all 0.
    # in simple words, y will be of shape (m, k)
    k = len(set(y))  # no. of class  (can also use np.unique)
    m = X_train.shape[0]  # no.of samples
    n = X_train.shape[1]  # no. of features
    Y_train_encoded = np.zeros((m, k))
    for each_class in range(k):
        # 判断向量y_train哪个位置和当前的类别相同，相同返回true，最终得到一个类别的bool向量
        cond = y_train == each_class
        # 将向量转化为矩阵
        Y_train_encoded[np.where(cond), each_class] = 1
    # Visualize our data
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], label='class 0', c=y)
    # plt.show()
    model = LogisticRegression(k, X_train.shape[1], "minibatch")
    model.fit(X_train, Y_train_encoded)
    yhat = model.predict(X_test)
    # model.plot()
    return plt,model,y_test,yhat,k


if __name__ == "__main__":
    plt,model,y_test,yhat,k = prep_report_data()
    plt.show()
    model.plot()
    print("=========Classification report=======")
    print("Report: ", classification_report(y_test, yhat))
    model.my_classification_report(y_test,yhat,k)

