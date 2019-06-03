'''
FU NLP-Beginner 
Task 2 : DL Text Classifier
Author : Lvxiaowei Xu
Set Up : 2019.5.25
'''

# 本文件代码用于完成Task2整个流程

#Py Import
import TextWashing as tw           #Step - 文本清洗
import Word2Vec as w2v             #Step - 词嵌入
import TextCNN_Train as TCNN       #Step - TextCNN训练
import TextRNN_Train as TRNN       #Step - TextRNN训练
import torch.utils.data as Data
#lib
import numpy             as np
import scipy.io as sio
import torch.utils.data as Data
import torch
from torch.autograd import Variable

if __name__ == '__main__':
    #============================================================================================================================
    # 实验1 TextCNN_随机词嵌入 
    #============================================================================================================================
    # 文本预处理 - 文本清洗
    TextWashingHandle      = tw.DLTextWashing()
    [p_Words,p_Label]      = TextWashingHandle.text_washing('/Users/xulvxiaowei/Documents/FDU NLP Tasks/Task2-DL_Text_Classifier/dataset/train2.tsv')
    # 随机词嵌入
    Vec_Dimension          = 100
    Classes                = 5
    Word2VecHandle         = w2v.WordEmbedding(p_Words,nclass = Classes,Dim = Vec_Dimension,InitMode = 'Random',rdvec = 1e-5)
    EmbededVec,EmLabel     = Word2VecHandle.embedding(p_Label,FreqLow = 10,Save= False)
    print(EmbededVec.size(),EmLabel.size())
    # 生成Batch数据集
    Batch_Size             = 500
    Workers                = 2
    Scale                  = 0
    EmYield,Vali_X,Vali_Y  = Word2VecHandle.batch_data_generate(EmbededVec,EmLabel,Batch_Size,Scale,Workers)   
    # TextCNN训练
    Epoch_Num              = 20
    TextCNNHandle          = TCNN.DLCNN_Train(Data_Batch    = EmYield,
                                              Epoch_Num     = Epoch_Num,
                                              Vali_X        = Vali_X,
                                              Vali_Y        = Vali_Y,
                                              Filter_Num    = 200,
                                              Filter_Size   = [1,2,3,4,5],
                                              Out_Dim       = Classes,Drop_out = 0.5,
                                              Class_Name    = ['0','1','2','3','4'],
                                              OPti_Mode     = 'Adam',
                                              Dim           = Vec_Dimension,
                                              Print_Control = 100)

    
    #============================================================================================================================
    # 实验2 TextCNN_Glove预训练词向量嵌入
    #============================================================================================================================
    # 文本预处理 - 文本清洗
    TextWashingHandle      = tw.DLTextWashing()
    [p_Words,p_Label]      = TextWashingHandle.text_washing('/Users/xulvxiaowei/Documents/FDU NLP Tasks/Task2-DL_Text_Classifier/dataset/train2.tsv')
    # 随机词嵌入
    Vec_Dimension          = 100
    Classes                = 5
    glove_path             = '/Users/xulvxiaowei/Downloads/glove.6B/glove.6B.100d.txt'
    Word2VecHandle         = w2v.WordEmbedding(p_Words,nclass = Classes,Dim = Vec_Dimension,InitMode = 'Glove',rdvec = 1e-5)
    EmbededVec,EmLabel     = Word2VecHandle.embedding(p_Label,FreqLow = 10,Save= False,glove_path = glove_path)
    print(EmbededVec.size(),EmLabel.size())
    # 生成Batch数据集
    Batch_Size             = 500
    Workers                = 2
    Scale                  = 0
    EmYield,Vali_X,Vali_Y  = Word2VecHandle.batch_data_generate(EmbededVec,EmLabel,Batch_Size,Scale,Workers)   
    # TextCNN训练
    Epoch_Num              = 10
    TextCNNHandle          = TCNN.DLCNN_Train(Data_Batch    = EmYield,
                                              Epoch_Num     = Epoch_Num,
                                              Vali_X        = Vali_X,
                                              Vali_Y        = Vali_Y,
                                              Filter_Num    = 200,
                                              Filter_Size   = [1,2,3,4,5],
                                              Out_Dim       = Classes,Drop_out = 0.5,
                                              Class_Name    = ['0','1','2','3','4'],
                                              OPti_Mode     = 'Adam',
                                              Dim           = Vec_Dimension,
                                              Print_Control = 1)
    
    #============================================================================================================================
    # 实验3 TextCNN_CBoW词向量嵌入训练
    #============================================================================================================================
    # 文本预处理 - 文本清洗
    TextWashingHandle      = tw.DLTextWashing()
    [p_Words,p_Label]      = TextWashingHandle.text_washing('/Users/xulvxiaowei/Documents/FDU NLP Tasks/Task2-DL_Text_Classifier/dataset/train2.tsv')
    # 随机词嵌入
    Vec_Dimension          = 100
    Classes                = 5
    Word2VecHandle         = w2v.WordEmbedding(p_Words,nclass = Classes,Dim = Vec_Dimension,InitMode = 'CBoW',rdvec = 1e-5)
    CBoW_Epoch_Num         = 500
    CBoW_N_Gram            = 2
    EmbededVec,EmLabel     = Word2VecHandle.embedding(p_Label,FreqLow = 10,Save= False,Epoch_Num = CBoW_Epoch_Num,N_gram = CBoW_N_Gram,Print_Control = 1000)
    print(EmbededVec.size(),EmLabel.size())
    # 生成Batch数据集
    Batch_Size             = 500
    Workers                = 2
    Scale                  = 0
    EmYield,Vali_X,Vali_Y  = Word2VecHandle.batch_data_generate(EmbededVec,EmLabel,Batch_Size,Scale,Workers)   
    # TextCNN训练
    Epoch_Num              = 500
    TextCNNHandle          = TCNN.DLCNN_Train(Data_Batch    = EmYield,
                                              Epoch_Num     = Epoch_Num,
                                              Vali_X        = Vali_X,
                                              Vali_Y        = Vali_Y,
                                              Filter_Num    = 200,
                                              Filter_Size   = [1,2,3,4,5],
                                              Out_Dim       = Classes,Drop_out = 0.5,
                                              Class_Name    = ['0','1','2','3','4'],
                                              OPti_Mode     = 'Adam',
                                              Dim           = Vec_Dimension,
                                              Print_Control = 10000)
    
    #============================================================================================================================
    # 实验4 TextCNN_SkipGram词向量嵌入训练
    #============================================================================================================================
    # 文本预处理 - 文本清洗
    TextWashingHandle      = tw.DLTextWashing()
    [p_Words,p_Label]      = TextWashingHandle.text_washing('/Users/xulvxiaowei/Documents/FDU NLP Tasks/Task2-DL_Text_Classifier/dataset/train2.tsv')
    # 随机词嵌入
    Vec_Dimension          = 100
    Classes                = 5
    Word2VecHandle         = w2v.WordEmbedding(p_Words,nclass = Classes,Dim = Vec_Dimension,InitMode = 'SkipGram',rdvec = 1e-5)
    CBoW_Epoch_Num         = 500
    CBoW_N_Gram            = 2
    EmbededVec,EmLabel     = Word2VecHandle.embedding(p_Label,FreqLow = 10,Save= False,Epoch_Num = CBoW_Epoch_Num,N_gram = CBoW_N_Gram,Print_Control = 1000)
    print(EmbededVec.size(),EmLabel.size())
    # 生成Batch数据集
    Batch_Size             = 500
    Workers                = 2
    Scale                  = 0
    EmYield,Vali_X,Vali_Y  = Word2VecHandle.batch_data_generate(EmbededVec,EmLabel,Batch_Size,Scale,Workers)   
    # TextCNN训练
    Epoch_Num              = 500
    TextCNNHandle          = TCNN.DLCNN_Train(Data_Batch    = EmYield,
                                              Epoch_Num     = Epoch_Num,
                                              Vali_X        = Vali_X,
                                              Vali_Y        = Vali_Y,
                                              Filter_Num    = 200,
                                              Filter_Size   = [1,2,3,4,5],
                                              Out_Dim       = Classes,Drop_out = 0.5,
                                              Class_Name    = ['0','1','2','3','4'],
                                              OPti_Mode     = 'Adam',
                                              Dim           = Vec_Dimension,
                                              Print_Control = 10000)
    #============================================================================================================================
    # 实验5 TextRNN_Glove预训练词向量嵌入
    #============================================================================================================================
    # 文本预处理 - 文本清洗
    TextWashingHandle      = tw.DLTextWashing()
    [p_Words,p_Label]      = TextWashingHandle.text_washing('/Users/xulvxiaowei/Documents/FDU NLP Tasks/Task2-DL_Text_Classifier/dataset/train2.tsv')
    # 随机词嵌入
    Vec_Dimension          = 100
    Classes                = 5
    glove_path             = '/Users/xulvxiaowei/Downloads/glove.6B/glove.6B.100d.txt'
    Word2VecHandle         = w2v.WordEmbedding(p_Words,nclass = Classes,Dim = Vec_Dimension,InitMode = 'Glove',rdvec = 1e-5)
    EmbededVec,EmLabel     = Word2VecHandle.embedding(p_Label,FreqLow = 10,Save= False,glove_path = glove_path)
    print(EmbededVec.size(),EmLabel.size())
    # 生成Batch数据集
    Batch_Size             = 500
    Workers                = 2
    Scale                  = 0
    EmYield,Vali_X,Vali_Y  = Word2VecHandle.batch_data_generate(EmbededVec,EmLabel,Batch_Size,Scale,Workers)   
    # TextCNN训练
    Epoch_Num              = 10
    TextCNNHandle          = TRNN.DLRNN_Train(Data_Batch    = EmYield,
                                              Epoch_Num     = Epoch_Num,
                                              Vali_X        = Vali_X,
                                              Vali_Y        = Vali_Y,
                                              Hidden_Size   = 200,
                                              Layers        = 2,
                                              Out_Dim       = Classes,
                                              bidirectional = True,
                                              Drop_out      = 0.5,
                                              Class_Name    = ['0','1','2','3','4'],
                                              OPti_Mode     = 'Adam',
                                              Dim           = Vec_Dimension,
                                              Print_Control = 1)
