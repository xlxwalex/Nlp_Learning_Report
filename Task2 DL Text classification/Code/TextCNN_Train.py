'''
FU NLP-Beginner 
Task 2 : DL Text Classifier
Author : Lvxiaowei Xu
Set Up : 2019.5.26
'''
# Sec(5.1)
# 本文件代码用于PyTorch的词嵌入操作，输入词语，返回词向量
# 代码注释中将包含与Report对应的公式编号或是节数
# 本部分完成TextCNN分类模型训练

import random
import time
import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn as nn
from   torch.autograd  import Variable
from   sklearn import metrics

# PyLib
import TextCNN  as TCNN                 #引入TextCNN类

# 训练TextCNN
class DLCNN_Train:
    # 初始化
    def __init__(self,Data_Batch,Vali_X,Vali_Y,Epoch_Num,Filter_Num,Filter_Size,Out_Dim,Class_Name,OPti_Mode = 'SGD',Drop_out = 0.5,Dim = 100,Print_Control = 1000,model_save ='-1'):
        '''
        Input
            Data_Batch      : 数据Batch生成器
            Vali_X          : 验证集的数据
            Vali_Y          : 验证集的标签
            Epoch_Num       : Epoch次数
            Filter_Num      : Filter_Num数量
            Filter_Size     : Filter Window大小
            Out_Dim         : 输出维度
            Class_Name      : 类别名称
            OPti_Mode       : 优化模型 SGD|Adam
            Dim             : Word2Vec词向量维度
            Drop_out        : DropOut参数
            Print_Control   : 打印控制参数
            model_save      : 模型存储位置
        '''
        # 初始化模型参数
        # Filter_Size : Default = [1,2,3,4,5]
        # Filter_Num  : Default = 200
        # Out_Dim     : Default = 5 
        self.Classes                = Class_Name
        # Model GPU
        self.device, self.gpucount  = self.get_device()
        if self.gpucount > 0:
            torch.cuda.manual_seed_all(2019)
            torch.backends.cudnn.deterministic = True 
        self.TextCNNModel = TCNN.TextCNN(Dim, Filter_Num, Filter_Size,Out_Dim, Drop_out)
        # 损失函数为交叉熵
        self.criterion    = nn.CrossEntropyLoss()
        # 是否送入GPU
        if self.gpucount > 0:
            self.TextCNNModel = self.TextCNNModel.to(self.device)
            self.criterion    = self.criterion.to(self.device)
        # 优化方式为SGD
        if OPti_Mode == 'SGD':
            self.optimizer    = optim.SGD(self.TextCNNModel.parameters(),lr=0.01)
        elif OPti_Mode == 'Adam':
            self.optimizer    = optim.Adam(self.TextCNNModel.parameters())
        else:
            print('Optimier Params Wrong')
            return 
        # 开始训练
        self.TextCNNModel.train()
        self.train(Data_Batch,Vali_X,Vali_Y,Epoch_Num,Print_Control,model_save)

    # 模型训练
    def train(self,Data_Batch,Vali_X,Vali_Y,Epoch_Num,Print_Control,model_save):
        self.Steps  = 0
        # 初始化Loss
        self.all_Loss     = []
        self.all_accuracy = []
        self.all_f1       = []
        for epoch in range(Epoch_Num):   
            print(f'---------------- Epoch: {epoch+1:02} ----------------')
            # 初始化参数
            all_preds     = np.array([], dtype=int)
            all_labels    = np.array([], dtype=int)
            epoch_loss    = 0
            train_steps   = 0
            for step,batch_data in enumerate(Data_Batch): 
                # 读取Batch数据并转换为Variable
                Data_X,Data_Y = batch_data
                Data_X,Data_Y = Variable(Data_X),Variable(Data_Y)
                # 初始化优化器的所有梯度               
                self.optimizer.zero_grad()
                # 得到Model的前向值
                if self.gpucount > 0:
                    CNNVal   = self.TextCNNModel(Data_X.to(self.device))
                else:
                    CNNVal   = self.TextCNNModel(Data_X)
                # 得到Loss值
                if self.gpucount > 0:
                    loss     = self.criterion(CNNVal,Data_Y.long().to(self.device))
                else:
                    loss     = self.criterion(CNNVal,Data_Y.long())
                epoch_loss  += loss.item()
                train_steps += 1
                #预测数据
                labels       = Data_Y.detach().cpu().numpy()
                preds        = np.argmax(CNNVal.detach().cpu().numpy(), axis=1)
                all_preds    = np.append(all_preds, preds)
                all_labels   = np.append(all_labels, Data_Y.long())
                #反向更新
                loss.backward()
                self.optimizer.step()
                self.Steps  += 1
                #打印输出
                if train_steps % Print_Control == 0:
                    print('Step{} Loss:{:.3f}'.format(train_steps,float(loss.detach().cpu().numpy())))
            # 计算模型性能
            '''
            if self.gpucount > 0:
                ValiVal  = self.TextCNNModel(Vali_X.to(self.device))
            else:
                ValiVal  = self.TextCNNModel(Vali_X)
            preds = np.argmax(ValiVal.detach().cpu().numpy(), axis=1)
            '''
            acc,f1_score = self.model_performance(all_preds,all_labels,self.Classes)
            self.all_Loss.append(epoch_loss)
            self.all_accuracy.append(acc)
            self.all_f1.append(f1_score)
            print('Epoch{} Acc:{:.3f} F1_Score:{:.3f}'.format(epoch+1,acc,f1_score))
        if model_save != '-1':
            torch.save(self.TextCNNModel.state_dict(), model_save + 'textcnn_model.pt')
            print('TextCNN已保存')
    
    # 性能分析   
    def model_performance(self,Data_Preds,Data_Labels,LabelLs):
        accuracy = metrics.accuracy_score(Data_Preds, Data_Labels)
        report = metrics.classification_report(Data_Preds, 
                                               Data_Labels, 
                                               digits = 5)
        F1_Avg = float(list(filter(None,report.split('\n')[-2].split(' ')))[5])
        #F1_Avg = float(list(filter(None,report.split('\n')[-3].split(' ')))[4])
        return accuracy,F1_Avg
    
    # 得到GPU信息
    def get_device(self):
        '''
        OutPut
            device     : 设备handle
            n_gpu      : GPU的个数
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n_gpu  = torch.cuda.device_count()
        if torch.cuda.is_available():
            print("device is cuda,  cuda is: ", n_gpu)
        else:
            print("device is cpu, GPU Forbidden")
        return device, n_gpu
