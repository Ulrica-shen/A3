import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import time
import copy
import pandas as pd
from collections import Counter


class LogisticRegression:

    def __init__(self, k, n, method, penalty=None, alpha=0.001, max_iter=5000):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.penalty = penalty  # 是否添加惩罚系数

    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []

        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad = self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")

        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0])  # <----with replacement
                batch_X = X[ix:ix + batch_size]
                batch_Y = Y[ix:ix + batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")

        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad

                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")

        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')

    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = - np.sum(Y * np.log(h)) / m
        error = h - Y
        grad = self.softmax_grad(X, error)
        if self.penalty is None:
            return loss, grad
        elif self.penalty == 'ridge':
            return loss+self.alpha * np.sum(np.square(h)),grad+self.alpha * 2 * h

    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return X.T @ error

    def h_theta(self, X, W):
        '''
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        '''
        return self.softmax(X @ W)

    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)

    def plot(self):
        plt.plot(np.arange(len(self.losses)), self.losses, label="Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()

    def my_classification_report(self,y_true,y_pred,k):

        # 得到混淆矩阵
        mcm = self.get_confusion_matrix(y_true,y_pred,k)
        # 得到TP,FN,FP,TN并且计算precision，recall，f1-score，support
        # 并且输出报告
        print("=========my Classification report=======")
        all_data = []
        for j in range(k):
            all_data.append(self.get_precision_recall_fscore_support(mcm,j))
        # 上半部分
        data_1 = {}
        label_1 = [i for i in range(k)] # 分类

        all_data_keys = ['precision_c', 'recall_c','f1_c', 'support',]
        cols_labels = ['precision', 'recall', 'f1_score','support']
        for label,col in zip(all_data_keys,cols_labels):
            data_1[col] = [item.get(label, 'error') for item in all_data]

        df_1 = pd.DataFrame(data_1,index=label_1)
        # df_1.drop(df_1.columns[0],axis=1,inplace=True)
        print(df_1)
        print()
        self.accuracy_macro_weighted(all_data,k,y_true)

    # 计算权重
    def weights_y_true(self, y_true , k):
        dic = dict(Counter(y_true))
        length = len(y_true)
        for key,val in dic.items():
            dic[key] = round(val/length,2)
        return dic

    def avg(self, all_data, label, k, weight=None):
        if all_data is None:
            raise ValueError(f'the parameters all_data could not be None')
        total = 0
        for data_item,idx in zip(all_data, range(k)):
            if data_item.get(label,None) is None:
                raise ValueError(f'could not find key {label} in items of all_data')
            else:
                total += data_item.get(label) if weight is None else data_item.get(label) * weight[idx]
        return round(total,2)

    # 预测准确度
    def accuracy(self,all_data,length_y_true):
        return round(sum([item.get('accuracy') for item in all_data]) / length_y_true,2)

    def accuracy_macro_weighted(self, all_data,k, y_true):
        # 下半部分
        data_2 = {}
        label_2 = ['accuracy', 'macro avg', 'weighted avg ']
        support = sum([item.get('support', 'error') for item in all_data])
        weights = self.weights_y_true(y_true, k)
        # 计算
        all_data_keys = ['precision_c', 'recall_c','f1_c']
        cols_labels = ['precision', 'recall', 'f1_score']

        for label,col in zip(all_data_keys,cols_labels):
            if label in ['precision_c', 'recall_c']:
                data_2[col] = ['']
            else:
                data_2[col] = [self.accuracy(all_data,len(y_true))]

            data_2[col].append(self.avg(all_data, label, k))
            data_2[col].append(self.avg(all_data, label, k, weights))

        data_2['support'] = [len(y_true)] * k
        df_2 = pd.DataFrame(data_2, index=label_2)
        print(df_2)

    # k 是分类种数
    def get_confusion_matrix(self, y_true,y_pred, k, sample_weight=1):
        # 统计每一种分类对应的预测分类情况
        # 当前分类被预测成 各分类的数量
        cate_pred = [0] * k
        matrix = []
        # 当前分类
        for cate in range(k):
            # 统计当前分类 被预测成 各分类 数量
            for idx in range(len(y_true)):
                # 找到当前分类的对应预测值，并修改数量
                if y_true[idx] == cate:
                    cate_pred[y_pred[idx]] += 1
            matrix.append(cate_pred)
            cate_pred = [0] * k

        return matrix

    def get_precision_recall_fscore_support(self, confusion_matrix, classification):
        # TP = True Positive: true is positive, predict is positive
        # FN = False Negative: true is positive, predict is negative
        # FP = False Positive: true is negative, predict is positive
        # TN = True Positive: true is negative, predict is negative

        # TP
        tp_c = confusion_matrix[classification][classification]

        # FN
        fn_list = confusion_matrix[classification].copy()
        del fn_list[classification]
        fn_c = sum(fn_list)

        # FP
        matrix = np.array(copy.deepcopy(confusion_matrix))
        fp_c = np.sum(matrix, axis=0)[classification]-tp_c

        precision_c = round(tp_c/(tp_c+fp_c),2)
        recall_c = round(tp_c/(tp_c+fn_c),2)
        f1_c = round(2 * precision_c * recall_c / (precision_c + recall_c),2)
        support = np.sum(matrix, axis=1)[classification]

        return {'precision_c': precision_c, 'recall_c': recall_c,
                'f1_c': f1_c, 'support': support, 'accuracy': tp_c}
