'''
FU NLP-Beginner 
Task 1 : Text Classifier
Author : Lvxiaowei Xu
Time   : 2019.5.13
'''
# 本文件代码用于完成模型部分Sec.5
# (包括Softmax模型、梯度下降参数学习、评价方法)
# 代码注释中将包含与Report对应的公式编号或是节数

#Lib
import numpy as np
import time
from random import shuffle
import math


#Softmax分类器类 Sec.5
class SoftmaxClassifier:
    #初始化操作
    def __init__(self,X, y_true, n_split,n_classes, n_iters=10, learning_rate=0.05, GD='BGD',Batch = -1):
        '''
        Input 
          X              : 训练集特征
          y_true         : 数据标签
          n_split        : 训练集与测试集的比例，为0时仅使用训练集
          n_classes      : 分类数目
          n_iters        : 迭代轮数
          learning_rate  : 学习率设置
          GD             : 优化算法('SGD','BGD','MBGD')
          Batch          : 当GD为MBGD时，需要通过Batch来指定划分为多少数据集
        '''
        print('Step3 : 模型训练准备中')
        self.n_samples, n_features = X.shape
        self.n_classes      = n_classes
        self.n_split        = n_split
        self.RandomGen      = np.random.RandomState(2019)
        self.weights        = self.RandomGen.rand(self.n_classes, n_features)
        self.bias           = np.ones((1, self.n_classes)) * 0.1
        self.all_losses     = []
        self.y_true         = y_true
        self.n_iters        = n_iters
        self.GD             = GD
        self.learning_rate  = learning_rate
        self.X              = X
        self.Batch          = Batch
        if n_split != 0:
            self.train_test_split()
        if Batch != -1:
            self.BatchSize   = math.floor(self.n_samples / (self.Batch))
            y_one_hot = self.one_hot(self.y_true)
            self.Batch_handle = self.batch_gen([self.X, y_one_hot],  self.BatchSize)
        print('Weights:{},Bias:{}'.format(self.weights.shape,self.bias.shape))
    
    #开始模型训练
    def fit(self):
        print('训练开始[Train Start]')
        if self.GD == 'SGD' or self.GD == 'MBGD':
            self.shuffle_data()
        y_one_hot = self.one_hot(self.y_true)
        for i in range(self.n_iters):
            if  self.GD == 'BGD':
                loss = self.softmax_BGD(self.X,self.learning_rate,y_one_hot)
                if i % 10 == 0:
                    print(f'Iteration number: {i}, loss: {np.round(loss, 8)}')
            elif self.GD == 'SGD':
                print('TURN:{}'.format(i))
                loss = self.softmax_SGD(self.X,self.learning_rate,y_one_hot)
            elif self.GD == 'MBGD':
                if self.Batch == -1:
                    print('未指定Batch大小,错误!')
                    return
                [Batch_X,Batch_Y] = next(self.Batch_handle)
                loss = self.softmax_MBGD(Batch_X,self.learning_rate,Batch_Y,y_one_hot)
                if i % 10 == 0:
                    print(f'Iteration number: {i}, loss: {np.round(loss, 8)}')
            else:
                print('ERROR: Params[GD] WRONG!')
                return 

        return self.weights, self.bias, self.all_losses
    
    #MBGD法进行优化
    def softmax_MBGD(self,X,learning_rate,y,Y_true):
        scores        = self.get_scores(X)
        probs         = self.softmax(scores)
        dw            = (1 / self.BatchSize) * np.dot(X.T, (probs - y))      #Eq.(5-16)
        db            = (1 / self.BatchSize) * np.sum(probs - y, axis=0)
        self.weights  = self.weights - learning_rate * dw.T
        self.bias     = self.bias - learning_rate * db
        #计算整体Loss
        scores        = self.get_scores(self.X)
        probs         = self.softmax(scores)
        loss          = self.cross_entropy(Y_true, probs)
        self.all_losses.append(loss)
        return loss

    #BGD法进行优化
    def softmax_BGD(self,X,learning_rate,y):
        scores        = self.get_scores(X)
        probs         = self.softmax(scores)
        loss          = self.cross_entropy(y, probs)
        self.all_losses.append(loss)
        dw            = (1 / self.n_samples) * np.dot(X.T, (probs - y))      #Eq.(5-16)
        db            = (1 / self.n_samples) * np.sum(probs - y, axis=0)
        self.weights  = self.weights - learning_rate * dw.T
        self.bias     = self.bias - learning_rate * db
        return loss

    #SGD法进行优化
    def softmax_SGD(self,X,learning_rate,y):
        for i in range(0,len(X)):
            scores        = self.get_scores(X[i])
            probs         = self.softmax(scores)
            #loss          = self.cross_entropy_SGD(y, probs)
            #self.all_losses.append(loss)
            dw            = np.dot(np.array([X[i]]).T, (probs - y[i]))      #Eq.(5-16)
            db            = np.sum(probs - y[i],axis=0)
            self.weights  = self.weights - learning_rate * dw.T
            self.bias     = self.bias - learning_rate * db
            if i % 100 == 0:
                scores        = self.get_scores(X)
                probs         = self.softmax(scores)
                loss          = self.cross_entropy(y, probs)
                self.all_losses.append(loss)
                print(f'Iteration number: {i}, loss: {np.round(loss, 8)}')
        return loss

    #模型预测
    def predict(self, X):
        '''
        Input 
          X              : 训练集数据
        Output
          np.argmax[]    : 预测标签one-hot
        '''
        scores = self.get_scores(X)
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)[:, np.newaxis]

    #计算Softmax输出
    def softmax(self, scores):
        '''
        Input 
          scores         : 训练集数据
        Output
          softmax        : softmax函数输出
        '''
        #Eq.(5-4)
        exp = np.exp(scores)
        sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
        softmax = exp / sum_exp
        return softmax

    #计算线性函数输出
    def get_scores(self, X):
        '''
        Input 
          X              : 训练集数据
        Output
          np.dot[]       : 总体的线性函数输出
        '''
        return np.dot(X, self.weights.T) + self.bias

    #计算交叉熵
    def cross_entropy(self, y_true, scores):
        '''
        Input 
          y_true         : 训练集标签数据
          scores         : 线性函数得分
        Output
          loss           : 总体的交叉熵损失
        '''
        loss = - np.sum(y_true * np.log(scores))      #Eq.(5-8)
        return loss

    #得到标签数据的
    def one_hot(self, y):
        '''
        Input 
          y              : 训练集标签数据
        Output
          one_hot        : 返回one-hot处理后的标签
        '''
        one_hot = np.zeros((self.n_samples, self.n_classes))
        one_hot[np.arange(self.n_samples), y.T] = 1
        return one_hot
    
    #计算Accuracy
    def get_accuracy(self,Mode = 'Tr'):
        '''
        Input 
          Mode           : 决定判断训练集还是测试集的Accuracy
        Output
          one_hot        : 返回one-hot处理后的标签
        '''
        if Mode == 'Tr':
            print('[Train Accuracy]')
            y_pred        = self.predict(self.X)
            y_predout     = np.reshape(y_pred,[1,len(y_pred)])[0]
            Accuracy      = np.sum(np.array(y_predout) == (np.array(self.y_true))) / self.n_samples * 100
        elif Mode == 'Te':
            print('[Test Accuracy]')
            y_pred        = self.predict(self.TeX)
            y_predout     = np.reshape(y_pred,[1,len(y_pred)])[0]
            Accuracy      = np.sum(np.array(y_predout) == (np.array(self.TeY))) / self.n_tests * 100
        else:
            print('ERROR:输入的Mode参数有误')
            Accuracy      = 0
        return Accuracy
    
    #对数据做shuffle
    def shuffle_data(self):
        index        = [i for i in range(self.n_samples)]
        self.RandomGen.shuffle(index)
        self.X       = self.X[index,:]
        self.y_true  = self.y_true[index]
    
    #对数据进行切分
    def train_test_split(self):
        index          = [i for i in range(self.n_samples)]
        self.RandomGen.shuffle(index)
        n_Train        = math.ceil(self.n_split * self.n_samples)
        self.n_tests   = self.n_samples - n_Train
        self.TeX       = self.X[index[n_Train:],:]
        self.TeY       = self.y_true[index[n_Train:]]
        self.X         = self.X[index[0:n_Train],:]
        self.y_true    = self.y_true[index[0:n_Train]]
        self.n_samples = n_Train
    
    #为MBGD产生Batch数据
    def batch_gen(self, XY , batch_size):
        '''
        Input 
          XY           : 整个数据集[训练集特征|标签]
          batch_size   : 每个batch的大小
        Output
          yield output
        '''
        XY = [np.array(data) for data in XY]
        data_size = XY[0].shape[0]
        batch_count = 0
        while True:
            if batch_count * batch_size + batch_size > data_size:
                batch_count = 0
            start = batch_count * batch_size
            end = start + batch_size
            batch_count += 1
            yield [data[start: end] for data in XY]
