'''
FU NLP-Beginner 
Task 1 : Text Classifier
Author : Lvxiaowei Xu
Time   : 2019.5.14
'''
# 本文件代码用于完成整个流程

#Py Import
import TextWashing as tw           #Step - 文本清洗
import Feature_Engineering as fe   #Step - 特征工程
import Model_Process as mp         #Step - 模型训练

#lib
import numpy             as np
import scipy.io as sio

[p_Words,p_Label]      = tw.text_washing('/Users/xulvxiaowei/Documents/FDU NLP Tasks/Task1-Text_Classifier/dataset/train.tsv')

p_combvec = fe.feature_main(p_Words,p_Label,Mode=False,LowFreq=20)

Tr_X, Tr_Y = p_combvec,np.array(p_Label)
Regressor = mp.SoftmaxClassifier(Tr_X,Tr_Y,n_split = 0.8,n_classes = 5,n_iters=3, learning_rate=0.1, GD='MBGD',Batch = 5)

[w_trained, b_trained, loss] = Regressor.fit()


print('Accuracy:',Regressor.get_accuracy(Mode = 'Te'))

#Save
sio.savemat('Loss_BGD.mat',{'loss':loss})

