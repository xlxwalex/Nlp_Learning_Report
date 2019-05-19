'''
FU NLP-Beginner 
Task 1 : Text Classifier
Author : Lvxiaowei Xu
Time   : 2019.5.12
'''
# 本文件代码用于特征工程 Sec.4
# (包括Bow模型以及Bi-gram模型)
# 代码注释中将包含与Report对应的公式编号或是节数

#Import Lib
import numpy as np
import time
import math
import scipy.io as sio

#词表创建 {Bow & N-gram}
def create_vocab(Words, FreqLow = 0):
    '''
    Input 
      Words        : 数据集单词表
      FreqLow      : 滤除频次小于FreqLow的Vocab字典
    Output
      p_IndexOut   : 单词对应的索引值
      p_VoCountOut : 单词出现的次数
    '''
    p_Index   = {}
    p_VoCount = {}
    try:
        for sentence in Words:
            for word in sentence:
                if word not in p_Index.keys():
                    p_Index[word] = len(p_Index.keys())
                    p_VoCount[p_Index[word]]  = 1
                else:
                    p_VoCount[p_Index[word]] += 1
    except:
        print('特征工程[生成词汇表]出现错误')
    #频次筛选
    p_IndexOut   = {}
    p_VoCountOut = {}
    for word in p_Index.keys():
        if p_VoCount[p_Index[word]] >= FreqLow:
            p_IndexOut[word] = len(p_IndexOut.keys())
            p_VoCountOut[p_IndexOut[word]] = p_VoCount[p_Index[word]]

    print('特征工程[生成词汇表]已完成')
    return p_IndexOut,p_VoCountOut
            
#Bow模型 Sec4.1
def Feature_BoW(Index,Words):
    '''
    Input 
      Index     : 数据集单词表
      Words     : 单词数据
    Output
      p_BowVec  : BoW特征对应的句子向量
    '''
    TimeCo = time.time()
    #得到向量空间大小
    Vec_Size = len(Index.keys())
    #得到总共向量的个数
    Vec_Count = len(Words)
    #生成Bow特征
    p_BowVec  = np.zeros((Vec_Count,Vec_Size),dtype=np.int)
    Sentence_Index = 0
    try:
        for Sentence in Words:
            for j in Sentence:
                if j in Index.keys():
                    p_BowVec[Sentence_Index,Index[j]] += 1
            Sentence_Index += 1
    except:
        print("特征工程[Bow特征提取]出现错误")
    print('特征工程[Bow特征提取]已完成,耗时{:.2f}s'.format(time.time()-TimeCo))
    return p_BowVec

#Bi-gram组合
def bigram_count(Index,VoCount,Words,FreqLow = 0):
    '''
    Input 
      Index        : 数据集单词表
      VoCount      : 单词对应的索引值
      Words        : 单词数据
    Output
      Bigram_Mat_T : Bi-gram的组合
      p_indexLen   : 筛选后的Bi-gram组合个数
    '''
    TimeCO        = time.time()
    Bigram_Size   = len(Index.keys())
    Bigram_Mat_T  = np.zeros((Bigram_Size,Bigram_Size),dtype=np.int)
    #计算频数
    for sentence in Words:
        if len(sentence) in [0,1]:
            continue
        for wdIx in range(1,len(sentence)):
            Bigram_Mat_T[Index[sentence[wdIx-1]],Index[sentence[wdIx]]] += 1
    #处理低频组合
    print(len(np.where(Bigram_Mat_T>FreqLow)[0]))
    gram_index  = 0
    Vocab_keys  = list(Index.keys())
    Bi_dict     = {}
    for Linedx in range(1,Bigram_Size):
        for Coldx in range(1,Bigram_Size):
            if Bigram_Mat_T[Linedx,Coldx] >= FreqLow:
                Bi_dict[(Vocab_keys[Linedx],Vocab_keys[Coldx])] = gram_index
                gram_index += 1
    p_indexLen = gram_index
    print('特征工程[Bi-gram组合生成]已完成')

    return Bi_dict,p_indexLen

#Bi-gram特征生成
def bigram_feature(Vocab,Index,Words,Bi_gramSize):
    '''
    Input 
      Vocab        : 词汇表
      Index        : Bi-gram组合单词表
      Words        : 单词
      Bi_gramSize  : Bi-gram组合个数
    Output
      p_BigramVec  : Bi-gram单词特征
    '''
    p_BigramVec  = np.zeros([len(Words),Bi_gramSize],dtype=int)
    print('Bi_gram_size:{}'.format(Bi_gramSize))
    SentenceID   = 0
    Vocab_dict   = Index.keys()
    for sentence in Words:
        if len(sentence) == 0:
            SentenceID += 1
            continue
        for idx in range(1,len(sentence)):
            if (sentence[idx-1],sentence[idx]) in Vocab_dict:
                p_BigramVec[SentenceID,Index[(sentence[idx-1],sentence[idx])]] += 1
        SentenceID += 1
    print(p_BigramVec)
    print('特征工程[Bi-gram特征]已完成')
    return p_BigramVec

#特征融合主方法 Sec.4
def feature_main(Words,Label,Mode = False,LowFreq=0):
    '''
    Input 
      Words       : 数据
      Label       : 数据标签
      Mode        : 特征选择(0-BoW 1-Bi_gram)
      LowFreq     : 过滤频率小于LowFreq的词汇
    Output
      p_combvec   : 处理后的向量
    '''
    if Mode == False:
        [p_Index,p_VoCount] = create_vocab(Words,FreqLow = LowFreq)
        p_BowVec   = Feature_BoW(p_Index,Words)
        p_combvec  = p_BowVec
        print('Index Vec:{}'.format(len(p_Index.keys())))
    else:
        [p_Index,p_VoCount]        = create_vocab(Words)
        [p_Bi_dict,p_gram_index] = bigram_count(p_Index,p_VoCount,Words,FreqLow = LowFreq)
        p_combvec                  = bigram_feature(p_Index,p_Bi_dict,Words,p_gram_index)
        print('Index Vec:{}'.format(p_gram_index))
    return p_combvec
