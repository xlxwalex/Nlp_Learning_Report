'''
FU NLP-Beginner 
Task 2 : DL Text Classifier
Author : Lvxiaowei Xu
Set Up : 2019.5.11
Modify : 2019.5.22  
'''
# 本文件代码用文本清洗
# (包括PreProcess、Normalization、Tokenization以及Stop Words的筛除)
# 本部分代码继续沿用Task1，只是做了部分改进，并封装为了类方便使用

#Lib
import numpy as np
import time

class DLTextWashing:
    #初始化操作
    def __init__(self):
        #英文常用停用词表 摘自https://blog.csdn.net/oYeZhou/article/details/83059359?utm_source=blogxgwz9
        self.stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']
    
    #读取数据集文件
    def read_file(self,Path):
        '''
        Input 
         Path    : 数据集存放的位置
        Output
         Data_x  : 数据
         Data_y  : 标签
        '''
        Content,Label = [],[]
        #读取文件
        with open(Path, 'r', encoding='utf-8', errors='ignore') as file_h:
            file_h.readline()  #去除首行
            for line in file_h:
                try:
                    tmpv = line.strip().split('\t')
                    Content.append(tmpv[2])
                    Label.append(eval(tmpv[3]))
                except:
                    pass
                    print('文本清洗[读取]读取词典出错')
        return Content,Label

    #数据预处理判断
    def pre_process(self,Content,Label):
        '''
        Input 
          Content   : 输入的数据
          Label     : 输入的标签
        Output
          Content   : 处理后的数据
          Label     : 处理后的标签
        '''
        tmpIndexs = []
        for i in range(0,len(Content)):
            try:
                if Content[i] == "":
                    tmpIndexs.append(i)
                if Label[i] > 4 or Label[i] < 0:
                    tmpIndexs.append(i)
            except:
                tmpIndexs.append(i)
                pass
        del Content[i]
        del Label[i]
        print('文本清洗[预处理]已完成:共[{}]个数据有错误'.format(len(tmpIndexs)))
        return Content,Label

    #标准化 
    def wash_normalize(self,Content):
        '''
        Input 
          Content   : 输入的数据
        Output
          p_Content : 处理后的数据
        '''
        p_Content = []
        for i in Content:
            p_Content.append(i.lower())
        print('文本清洗[标准化]已完成')
        return p_Content

    #分词 
    def wash_tokenization(self,Content):
        '''
        Input 
          Content   : 输入的数据
        Output
          p_Words   : 处理后的数据
        '''
        p_Words = []
        for i in Content:
            #标点符号转换
            tmp_Content = i
            for j in [',','.','?','!',':','&']:
                tmp_Content = str.replace(tmp_Content,j,' ')
            #分词
            tmp_wdls = tmp_Content.split(' ')
            #去除空字符
            tmp_wdls = list(filter(None, tmp_wdls))
            p_Words.append(tmp_wdls)
        print('文本清洗[分词]已完成')
        return p_Words  

      #停用词筛除
    def wash_stopwords(self,Words):
        '''
        Input 
          Words     : 分词结果
        Output
          p_Words   : 处理后的结果
        '''
        for i in range(0,len(Words)):
            for j in range(len(Words[i])-1,-1,-1):
                if Words[i][j] in self.stoplist:
                    del Words[i][j]
            
        print('文本清洗[停用词筛除]已完成')
        return Words 

    #文本清洗主方法
    def text_washing(self,Path):
        '''
        Input 
          Path      : 数据集存放的位置
        Output
          p_Words   : 数据
          p_Label   : 标签
        '''
        print('Step1 : 文本清洗')
        TimeCO       = time.time()
        [C,L]        = self.read_file(Path)
        [C2,p_Label] = self.pre_process(C,L)
        C3           = self.wash_normalize(C2)
        P1           = self.wash_tokenization(C3)
        #p_Words     = wash_stopwords(P1)
        print('文本清洗完成，共计用时{:.2f}s'.format(time.time()-TimeCO))
        print("")
        return P1,p_Label