'''
FU NLP-Beginner 
Task 2 : DL Text Classifier
Author : Lvxiaowei Xu
Set Up : 2019.5.23
'''
# Sec.(4)
# 本文件代码用于PyTorch的词嵌入操作，输入词语，返回词向量
# 代码注释中将包含与Report对应的公式编号或是节数
# 本部分完成词嵌入的过程

# Py Lib
import w2v.CBoW
import w2v.SkipGram

# Import Lib
import numpy as np
import time
import torch.nn as nn
import torch
from   torch.autograd import Variable
import scipy.io as sio
import torch.utils.data as Data
import math
from torchtext import data,datasets,vocab

# Word Embedding
class WordEmbedding:
    # 初始化操作
    def __init__(self,Words,nclass,Dim=100,InitMode = 'Random',rdvec = 1e-5):
        '''
        Input
            Words      : 预处理完的语料词语
            nclass     : 分类种类数目
            Dim        : Word2Vec词向量维度
            InitMode   : 初始化方式-'Random'随机初始化|'Glove'加载预训练的Glove向量|'CBoW'自动训练得到CBoW词嵌入向量|'SkipGram'自动训练得到SkipGram词嵌入向量
            rdvec      : 词嵌入平滑化系数
            glove_path : Glove预训练词向量存储位置
        '''
        self.Words      = Words
        self.Dim        = Dim
        self.n_classes  = nclass
        self.rdvec      = rdvec   
        # 指定随机嵌入种子
        torch.manual_seed(2019)
        self.RandomGen  = np.random.RandomState(2019)
        if InitMode   == 'Random':
            self.Mode = 0
        elif InitMode == 'Glove':
            self.Mode = 1
        elif InitMode == 'CBoW':
            self.Mode = 2
        elif InitMode == 'SkipGram':
            self.Mode = 3
        else:
            self.Mode = -1
            print('Word2Vec[词向量产生]参数初始化选择错误')

    # 词嵌入   
    def embedding(self,LabelIn,FreqLow = 0,Save = False,glove_path = '-1',Epoch_Num = 100,N_gram = 2,Print_Control = 10000,EmbVec_path = ''):
        '''
        Input
            LabelIn         : 输入标签
            FreqLow         : 低频词汇过滤阈值
            Save            : 是否存储
            glove_path      : Glove词向量存储地址
            Epoch_Num       : Epoch个数
            N_gram          : CBoW以及Skip-gram的上下文长度
            Print_Control   : 打印控制
            EmbVec_path     : CBoW词嵌入存储位置
        Output
            self.wordvec    : 返回语料的词向量矩阵
        '''
        # 判断词嵌入矩阵产生方式
        if self.Mode == 1:
            # 加载Glove向量
            self.glove_embedding(LabelIn,glove_path,FreqLow,False)
            WordVecVariables = Variable(self.wordvec,requires_grad = False)
            WordVecLabelVar  = Variable(torch.Tensor(self.wd_label),requires_grad = False)
            return WordVecVariables,WordVecLabelVar
        elif self.Mode == 0:
            # 随机初始化
            self.random_embedding(LabelIn,FreqLow)
            WordVecVariables = Variable(self.wordvec,requires_grad = False)
            WordVecLabelVar  = Variable(torch.Tensor(self.wd_label),requires_grad = False)
            return WordVecVariables,WordVecLabelVar
        elif self.Mode == 2:
            # CBoW词嵌入训练
            self.cbow_embedding(Epoch_Num,LabelIn,self.Dim,N_gram,FreqLow,Print_Control,'',False)
            WordVecVariables = Variable(self.wordvec,requires_grad = False)
            WordVecLabelVar  = Variable(torch.Tensor(self.wd_label),requires_grad = False)
            return WordVecVariables,WordVecLabelVar
        elif self.Mode == 3:
            # SkipGram词嵌入训练
            self.skipgram_embedding(Epoch_Num,LabelIn,self.Dim,N_gram,FreqLow,Print_Control,'',False)
            WordVecVariables = Variable(self.wordvec,requires_grad = False)
            WordVecLabelVar  = Variable(torch.Tensor(self.wd_label),requires_grad = False)
            return WordVecVariables,WordVecLabelVar
        else: #E RROR
            print('[Word2Vec]在[Embedding]出现错误')
    
    # CBoW词嵌入训练 Sec4.3
    def cbow_embedding(self,Epoch_Num,LabelIn,N_dim,N_gram,FreqLow = 0,Print_Control = 10000,EmbVec_path = '',Save =False):
        '''
        Input
            Epoch_Num       : Epoch个数
            LabelIn         : 输入标签
            glove_path      : 预训练的Glove词向量存储位置
            FreqLow         : 低频词汇过滤阈值
            Save            : 是否存储
            Print_Control   : 打印控制
            EmbVec_path     : CBoW词嵌入存储位置
        '''
        TmpTimeCo = time.time()
        wd_Index,wd_Freq,wd_VocabLen,wd_SenWordMax,wd_SentenceLen,self.wd_label = self.create_vocab(LabelIn,0)
        # CBoW模型训练
        CBoWTrainHandle = w2v.CBoW.DL_CBoW_Train(Words         = self.Words,
                                                 Vocab         = wd_Index,
                                                 Dim           = N_dim,
                                                 Epoch_Num     = Epoch_Num,
                                                 N_gram        = N_gram,
                                                 Print_Control = Print_Control,
                                                 EmbVec_path   = EmbVec_path
                                                )
        wd_Index_FreqForbidden,wd_Freq,wd_VocabLen,wd_SenWordMax,wd_SentenceLen,self.wd_label = self.create_vocab(LabelIn,FreqLow)
        self.embedMat = CBoWTrainHandle.CBoWModel.embedding.weight
        # 创建Word2Vec
        print('Word2Vec Size Params:',end = '')
        print(wd_SentenceLen,wd_SenWordMax,self.Dim)
        self.wordvec   = torch.zeros(wd_SentenceLen,wd_SenWordMax,self.Dim)
        # CBoW词嵌入   
        sentence_index = 0 
        for sentence in self.Words:
            word_index = 0
            if sentence == []:
                continue
            for word in sentence:
                if word in wd_Index_FreqForbidden.keys():
                    self.wordvec[sentence_index,word_index] = self.embedMat[wd_Index[word]]
                else:
                    # 词嵌入平滑化(当词表中查不到词汇时使用)
                    self.wordvec[sentence_index,word_index] = torch.rand(1, self.Dim) * self.rdvec
                word_index += 1
            sentence_index += 1
        # 返回语料CBoW词嵌入的嵌入矩阵
        print('词嵌入过程[WordVec生成(CBoW)]已完成,共计用时{:.2f}s'.format(time.time()-TmpTimeCo))
        # 存储词嵌入向量
        if Save:
            sio.savemat('Word2Vec_CBoW.mat',{'w2v':self.wordvec.numpy()})
            print('词嵌入过程[WordVec保存]已完成,文件名Word2Vec_CBoW.mat')
    
    # SkipGram词嵌入训练 Sec4.4
    def skipgram_embedding(self,Epoch_Num,LabelIn,N_dim,N_gram,FreqLow = 0,Print_Control = 10000,EmbVec_path = '',Save =False):
        '''
        Input
            Epoch_Num       : Epoch个数
            LabelIn         : 输入标签
            glove_path      : 预训练的Glove词向量存储位置
            FreqLow         : 低频词汇过滤阈值
            Save            : 是否存储
            Print_Control   : 打印控制
            EmbVec_path     : SkipGram词嵌入存储位置
        '''
        TmpTimeCo = time.time()
        wd_Index,wd_Freq,wd_VocabLen,wd_SenWordMax,wd_SentenceLen,self.wd_label = self.create_vocab(LabelIn,0)
        # SkipGram模型训练
        SkipGramTrainHandle = w2v.SkipGram.DL_SkipGramTrain(Words          = self.Words,
                                                             Vocab         = wd_Index,
                                                             Dim           = N_dim,
                                                             Epoch_Num     = Epoch_Num,
                                                             N_gram        = N_gram,
                                                             Print_Control = Print_Control,
                                                             EmbVec_path   = EmbVec_path
                                                             )
        wd_Index_FreqForbidden,wd_Freq,wd_VocabLen,wd_SenWordMax,wd_SentenceLen,self.wd_label = self.create_vocab(LabelIn,FreqLow)
        self.embedMat = SkipGramTrainHandle.SkipGramModel.embedding.weight
        # 创建Word2Vec
        print('Word2Vec Size Params:',end = '')
        print(wd_SentenceLen,wd_SenWordMax,self.Dim)
        self.wordvec   = torch.zeros(wd_SentenceLen,wd_SenWordMax,self.Dim)
        # SkipGram词嵌入   
        sentence_index = 0 
        for sentence in self.Words:
            word_index = 0
            if sentence == []:
                continue
            for word in sentence:
                if word in wd_Index_FreqForbidden.keys():
                    self.wordvec[sentence_index,word_index] = self.embedMat[wd_Index[word]]
                else:
                    # 词嵌入平滑化(当词表中查不到词汇时使用)
                    self.wordvec[sentence_index,word_index] = torch.rand(1, self.Dim) * self.rdvec
                word_index += 1
            sentence_index += 1
        # 返回语料SkipGram词嵌入的嵌入矩阵
        print('词嵌入过程[WordVec生成(SkipGram)]已完成,共计用时{:.2f}s'.format(time.time()-TmpTimeCo))
        # 存储词嵌入向量
        if Save:
            sio.savemat('Word2Vec_SkipGram.mat',{'w2v':self.wordvec.numpy()})
            print('词嵌入过程[WordVec保存]已完成,文件名Word2Vec_SkipGram.mat')

    # Glove预训练词向量嵌入 Sec4.2
    def glove_embedding(self,LabelIn,glove_path,FreqLow = 0,Save =False):
        '''
        Input
            LabelIn         : 输入标签
            glove_path      : 预训练的Glove词向量存储位置
            FreqLow         : 低频词汇过滤阈值
            Save            : 是否存储
        '''
        TmpTimeCo = time.time()
        wd_Index,wd_Freq,wd_VocabLen,wd_SenWordMax,wd_SentenceLen,self.wd_label = self.create_vocab(LabelIn,FreqLow)
        vocab_index = self.glove_import(glove_path)
        self.Dim = self.embedMat.size()[1]
        # 创建Word2Vec
        print('Word2Vec Size Params:',end = '')
        print(wd_SentenceLen,wd_SenWordMax,self.Dim)
        self.wordvec   = torch.zeros(wd_SentenceLen,wd_SenWordMax,self.Dim)
        index_key = vocab_index.keys()
        # Glove预训练的词嵌入   
        sentence_index = 0 
        for sentence in self.Words:
            word_index = 0
            if sentence == []:
                continue
            for word in sentence:
                if word in index_key:
                    self.wordvec[sentence_index,word_index] = self.embedMat[vocab_index[word]]
                else:
                    # 词嵌入平滑化(当词表中查不到词汇时使用)
                    self.wordvec[sentence_index,word_index] = torch.rand(1, self.Dim) * self.rdvec
                word_index += 1
            sentence_index += 1
        # 返回语料Glove预训练的嵌入矩阵
        print('词嵌入过程[WordVec生成(Glove)]已完成,共计用时{:.2f}s'.format(time.time()-TmpTimeCo))
        # 存储词嵌入向量
        if Save:
            sio.savemat('Word2Vec.mat',{'w2v':self.wordvec.numpy()})
            print('词嵌入过程[WordVec保存]已完成,文件名Word2Vec.mat')

    def glove_import(self,glove_path):
        '''
        Input
            glove_path      : 输入标签
        Output
            Vocab_Corpus    : 单词与向量之间的索引
            self.embedMat   : 读取的词向量权重
        '''
        TmpTimeCo      = time.time()
        Vocab_Corpus   = {}
        pretrained_Vec = []
        Vec_index      = 0
        with open(glove_path, 'r', encoding='utf-8', errors='ignore') as file_h:
            for line in file_h:
                try:
                    Vocab_Corpus[line.split(' ')[0]] = Vec_index 
                    tmpVec     = [float(i) for i in line.split(' ')[1:]]
                    pretrained_Vec.append(tmpVec)
                    Vec_index += 1
                except:
                    print('Word2Vec[Glove词向量]读取出错')
        self.embedMat = torch.Tensor(pretrained_Vec)
        print('词嵌入过程[Glove词向量]读取已完成,共计用时{:.2f}s'.format(time.time()-TmpTimeCo))
        return Vocab_Corpus

    # 随机初始化方式嵌入 Sec4.1
    def random_embedding(self,LabelIn,FreqLow = 0,Save = False):
        '''
        Input
            LabelIn        : 输入标签
            FreqLow        : 低频词汇过滤阈值
            Save           : 是否存储
        '''
        TmpTimeCo = time.time()
        wd_Index,wd_Freq,wd_VocabLen,wd_SenWordMax,wd_SentenceLen,self.wd_label = self.create_vocab(LabelIn,FreqLow)
        self.embedMat  = nn.Embedding(wd_VocabLen,self.Dim)
        # 创建Word2Vec
        print('Word2Vec Size Params:',end = '')
        print(wd_SentenceLen,wd_SenWordMax,self.Dim)
        self.wordvec   = torch.zeros(wd_SentenceLen,wd_SenWordMax,self.Dim)
        # 随机词嵌入   
        sentence_index = 0 
        for sentence in self.Words:
            word_index = 0
            if sentence == []:
                continue
            for word in sentence:
                if word in wd_Index.keys():
                    tmpTensor   = torch.LongTensor([wd_Index[word]])
                    tmpVariable = Variable(tmpTensor,requires_grad = False)
                    self.wordvec[sentence_index,word_index] = self.embedMat(tmpVariable)
                else:
                    # 词嵌入平滑化(当词表中查不到词汇时使用)
                    self.wordvec[sentence_index,word_index] = torch.rand(1, self.Dim) * self.rdvec
                word_index += 1
            sentence_index += 1
        # 返回语料的随机词嵌入矩阵
        print('词嵌入过程[WordVec生成(随机)]已完成,共计用时{:.2f}s'.format(time.time()-TmpTimeCo))
        # 存储词嵌入向量
        if Save:
            sio.savemat('Word2Vec.mat',{'w2v':self.wordvec.numpy()})
            print('词嵌入过程[WordVec保存]已完成,文件名Word2Vec.mat')

    # 词表创建
    def create_vocab(self, LabelIn,FreqLow = 0):
        '''
        Input 
            FreqLow      : 滤除频次小于FreqLow的Vocab字典
            LabelIn      : 数据集的Label标签
        Output
            p_IndexOut   : 单词对应的索引值
            p_VoCountOut : 单词出现的次数
            p_VocabLen   : 返回词表的大小
            p_WordLenMax : 返回语料中最大的词长
            p_Sentence   : 返回语料数量
            p_LabelOut   : 返回过滤后的标签数据
        '''
        p_Index    = {}
        p_VoCount  = {}
        MaxWordLen = 0
        SentenceC  = 0
        SentenceRe = 0
        p_LabelOut = []
        try:
            for sentence in self.Words:
                if sentence:
                    SentenceC += 1
                    p_LabelOut.append(LabelIn[SentenceRe])
                tmpsenmax = 0
                for word in sentence:
                    tmpsenmax += 1
                    if word not in p_Index.keys():
                        p_Index[word] = len(p_Index.keys())
                        p_VoCount[p_Index[word]]  = 1
                    else:
                        p_VoCount[p_Index[word]] += 1
                if tmpsenmax > MaxWordLen:
                    MaxWordLen = tmpsenmax
                SentenceRe += 1
        except:
            print('词嵌入过程[生成词汇表]出现错误')
        # 频次筛选
        p_IndexOut   = {}
        p_VoCountOut = {}
        for word in p_Index.keys():
            if p_VoCount[p_Index[word]] >= FreqLow:
                p_IndexOut[word] = len(p_IndexOut.keys())
                p_VoCountOut[p_IndexOut[word]] = p_VoCount[p_Index[word]]
        # 得到词表的大小及句子的单词最大长度
        p_VocabLen     = len(p_IndexOut.keys())
        p_WordLenMax   = MaxWordLen
        p_sentence     = SentenceC
        self.n_samples = p_sentence
        print('词嵌入过程[生成词汇表]已完成')
        return p_IndexOut,p_VoCountOut,p_VocabLen,p_WordLenMax,p_sentence,p_LabelOut
    
    # 得到标签数据One-Hot
    def one_hot(self, y):
        '''
        Input 
          y              : 训练集标签数据
        Output
          one_hot        : 返回one-hot处理后的标签
        '''
        one_hot = np.zeros((self.n_samples, self.n_classes))
        one_hot[np.arange(self.n_samples), y.T] = 1
        one_hot = torch.Tensor(one_hot)
        one_hot = Variable(one_hot,requires_grad = False)
        print('训练数据处理[One-Hot转换]已完成')
        return one_hot
    
    # 生成Batch数据集
    def batch_data_generate(self,EmbededVec,EmOHLabel,Batch_Size = 500,scale = 0.1,Workers = 2):
        '''
        Input 
          EmbededVec     : 训练集数据
          EmOHLabel      : 训练集标签数据
          Batch_Size     : Batch的大小
          Workers        : 同时工作的线程
        Output
          Data_Yield     : 返回可迭代数据
          Vali_X         : 验证集数据
          Vali_Y         : 验证集标签
        '''
        tmpTimeCo          = time.time()
        # 数据集拆分
        Train_X,Train_Y,Vali_X,Vali_Y = self.train_test_split(EmbededVec,EmOHLabel,scale)
        print('Train Size : {},Validation Size : {},Scale = {}'.format(Train_X.size()[0],Vali_X.size()[0],scale))
        # 生成Batch数据
        Batch_DataSet      = Data.TensorDataset(Train_X,torch.Tensor(Train_Y))
        Data_Yield         = Data.DataLoader(dataset = Batch_DataSet,batch_size = Batch_Size,shuffle = True,num_workers = Workers)
        print('训练数据处理[Batch生成]已完成')
        return Data_Yield,Vali_X,Vali_Y
    
    # 拆分训练集和验证集
    def train_test_split(self,DataIn,LabelIn,scale = 0.1):
        '''
        Input 
          DataIn         : 全部Vec数据
          LabelIn        : 全部标签数据
          scale          : Vali占比
        Output
          Data_Train_X   : 训练集数据
          Data_Train_Y   : 训练集标签
          Data_Vali_X    : 验证集数据
          Data_Vali_Y    : 验证集标签
        '''
        TotalVec           = DataIn.size()[0]
        index              = [i for i in range(TotalVec)]
        n_train            = math.ceil(TotalVec * (1-scale))
        self.RandomGen.shuffle(index)
        Data_Train_X       = DataIn [0:n_train,:,:]
        Data_Train_Y       = LabelIn[0:n_train]
        Data_Vali_X        = DataIn [n_train:,:,:]
        Data_Vali_Y        = LabelIn[n_train:]
        return Data_Train_X,Data_Train_Y,Data_Vali_X,Data_Vali_Y