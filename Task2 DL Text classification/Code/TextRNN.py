'''
FU NLP-Beginner 
Task 2 : DL Text Classifier
Author : Lvxiaowei Xu
Set Up : 2019.5.25
'''
# Sec(5.2)
# 本文件代码用于PyTorch的词嵌入操作，输入词语，返回词向量
# 代码注释中将包含与Report对应的公式编号或是节数
# 本部分完成TextRNN分类

import torch
import torch.nn as nn
import torch.nn.functional as F

# TextCNN的线性部分
class DLLinear(nn.Module):
    def __init__(self, Text_InFeature, Text_OutFeature):
        super(DLLinear, self).__init__()
        '''
        Input
            Text_InFeature  : 输入大小
            Text_OutFeature : 输出大小
        '''
        self.linear = nn.Linear(in_features    =  Text_InFeature,
                                out_features   =  Text_OutFeature
                                )
        # 初始化线性参数
        self.init_params()

    # 初始化线性参数
    def init_params(self):
        # 线性变换函数的初始化方法
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
    
    # 前向计算
    def forward(self, x):
        x = self.linear(x)
        return x

# TextRNN的LSTM部分
class DLLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout):
        super(DLLSTM, self).__init__()
        '''
        Input
            input_size      : 输入大小
            hidden_size     : 隐藏h的尺寸
            num_layers      : 层数
            bidirectional   : 是否双向
            dropout         : Dropout参数
        '''
        self.rnn = nn.LSTM(input_size    = input_size, 
                           hidden_size   = hidden_size,
                           num_layers    = num_layers, 
                           bidirectional = bidirectional, 
                           dropout       = dropout
                          )


    def forward(self, x):
        '''
        Input
            x               : 输入
        '''
        out_put, (hidden, cell) = self.rnn(x)
        return hidden, out_put

# TextRNN主类
class TextRNN(nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, dropout):
        super(TextRNN, self).__init__()
        '''
        Input
            embedding_dim   : 词向量维度
            output_dim      : 输出尺寸
            num_layers      : 层数
            bidirectional   : 是否双向
            dropout         : Dropout参数
        '''
        self.text_len = text_len
        self.rnn = DLLSTM(embedding_dim, hidden_size, num_layers,bidirectional, dropout)
        self.fc = DLLinear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        '''
        Input
            x               : 输入
        '''
        embedded = self.dropout(x)
        embedded = embedded.permute(1, 0, 2)
        hidden, outputs = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  
        return self.fc(hidden)