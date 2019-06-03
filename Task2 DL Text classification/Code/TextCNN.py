'''
FU NLP-Beginner 
Task 2 : DL Text Classifier
Author : Lvxiaowei Xu
Set Up : 2019.5.25
'''

# Sec(5.1)
# 本文件代码用于PyTorch的词嵌入操作，输入词语，返回词向量
# 代码注释中将包含与Report对应的公式编号或是节数
# 本部分完成TextCNN分类

import torch
import torch.nn as nn
import torch.nn.functional as F

# TextCNN的卷积层类
class DLConv1d(nn.Module):
    def __init__(self, Text_InChannels, Text_OutChannels, Filters,InitBias = 0.1):
        super(DLConv1d, self).__init__()
        '''
        Input
            Text_InChannels : 输入大小
            Text_OutChannels: 输出大小
            Filters         : 卷积核尺寸
            InitBias        : 初始化偏置
        '''
        # 根据Filter创建多个卷积层实例
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels  = Text_InChannels,
                      out_channels = Text_OutChannels,
                      kernel_size  = filtern
                    )
            for filtern in Filters
        ])
        # 初始化卷积层参数
        self.init_params(InitBias)

    # 初始化卷积层参数
    def init_params(self,InitBias):
        '''
        Input
            InitBias        : 初始化偏置
        '''
        for conv_filter in self.convs:
            # 针对Relu激活函数的初始化方法
            # Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification He, K. et al. (2015)
            nn.init.kaiming_uniform_(conv_filter.weight.data,a=0, mode='fan_in', nonlinearity='leaky_relu')
            # 将偏置初始化为常数
            nn.init.constant_(conv_filter.bias.data, InitBias)

    # 前向计算
    def forward(self, x):
        '''
        Input
            x                : 输入
        '''
        return [F.relu(convn(x)) for convn in self.convs]

# TextCNN的线性部分
class DLLinear(nn.Module):
    def __init__(self, Text_InFeature, Text_OutFeature):
        '''
        Input
            Text_InFeature  : 输入大小
            Text_OutFeature : 输出大小
        '''
        super(DLLinear, self).__init__()
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

# TextCNN主类
class TextCNN(nn.Module):
    
    def __init__(self,embedding_dim, n_filters, filter_sizes, output_dim,dropout):
        '''
        Input
            embedding_dim   : 词向量嵌入维度
            n_filters       : 卷积核尺寸
            filter_sizes    : 卷积核每种尺寸数量
            output_dim      : 输出维度
            dropout         : DropOut参数
        '''
        super().__init__()

        self.convs     = DLConv1d(embedding_dim,n_filters,filter_sizes)
        self.fc        = DLLinear(len(filter_sizes) * n_filters, output_dim)
        self.dropout   = nn.Dropout(dropout)

    # 前向计算
    def forward(self, embedded):
        '''
        Input
            embedd          : 训练数据词向量
        '''
        embedded = embedded.permute(0, 2, 1) 
        conved = self.convs(embedded)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

