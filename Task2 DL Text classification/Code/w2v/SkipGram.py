'''
FU NLP-Beginner 
Task 2 : DL Text Classifier
Author : Lvxiaowei Xu
Set Up : 2019.5.27
'''

# Sec(4.4)
# 本部分为Task2的拓展部分内容 - 关于词嵌入[Word2Vec]中SkipGram模型的训练
# 本文件代码用于PyTorch的词嵌入操作，输入词语，返回词向量
# 代码注释中将包含与Report对应的公式编号或是节数

# Import Lib
import torch
from   torch import nn,optim
import torch.nn.functional as F
from   torch.autograd import Variable
import scipy.io as sio
import numpy as np

# SkipGram 模型
class DL_SkipGram(nn.Module):
    # SkipGram模型参数初始化
    def __init__(self, N_word, N_dim, N_gram):
        super(DL_SkipGram, self).__init__()
        '''
        Input
            N_word     : 词的数量
            N_dim      : 词嵌入维度
            N_gram     : 上下文关系数
        '''
        self.embedding    = nn.Embedding(N_word, N_dim)
        self.linearlayer  = nn.Linear(in_features  = N_dim,
                                      out_features = N_word
                                     )
        self.log_softmax  = nn.LogSoftmax(dim=1)

    # 前向传播
    def forward(self, x):
        x = self.embedding(x)
        x = self.linearlayer(x)
        #x = F.log_softmax(x,dim = 1)
        x = self.log_softmax(x)
        return x

# SkipGram模型训练
class DL_SkipGramTrain():
    # SkipGram模型训练初始化
    def __init__(self,Words,Vocab,Dim = 100,Epoch_Num = 100,N_gram = 2,Print_Control = 10000,EmbVec_path = ''):
        '''
        Input
            Words          : 训练的语料
            Vocab          : 语料字典
            Dim            : 词嵌入维度
            Epoch_Num      : Epoch个数
            N_gram         : 上下文关系数
            Print_Control  : 打印控制
            EmbVec_path    : SkipGram词嵌入存储位置
        '''
        # 初始化训练数据
        self.DataIn     = []
        # SHuffle数据集
        self.RandomGen  = np.random.RandomState(2019)
        shuffle_index   = [i for i in range(0,len(Words))]
        self.RandomGen.shuffle(shuffle_index)
        # 生成输入层数据
        for sentenceID in shuffle_index:
            sentence    = Words[sentenceID]
            for i in range(N_gram, len(sentence) - N_gram):
                ContextLs = [sentence[j] for j in range(i - 2,i + 3)]
                Target    = sentence[i]
                for Context in ContextLs:
                    self.DataIn.append((Context, Target))
        print('SkipGram词嵌入训练[语料生成]完成,Size : {}'.format(len(self.DataIn)))
        # SkipGram模型初始化
        self.SkipGramModel  = DL_SkipGram(len(Vocab),Dim,N_gram)
        self.criterion      = nn.NLLLoss()
        self.optimizer      = optim.SGD(self.SkipGramModel.parameters(), lr=1e-3)
        # Model GPU
        self.device, self.gpucount  = self.get_device()
        # 使成绩结果统一
        if self.gpucount > 0:
            torch.cuda.manual_seed_all(2019)
            torch.backends.cudnn.deterministic = True 
        # 是否送入GPU
        if self.gpucount > 0:
            self.SkipGramModel.to(self.device)
            self.criterion.to(self.device)
        # 开始训练
        self.SkipGramModel.train()
        self.train(Vocab,Epoch_Num,Print_Control,EmbVec_path)
  
    # SkipGram模型训练
    def train(self,Vocab,Epoch_Num,Print_Control,EmbVec_path):
        '''
        Input
            Vocab         : 语料字典
            Epoch_Num     : Epoch个数
            Print_Control : 打印控制
            EmbVec_path   : 词嵌入保存地址
        '''
        for epoch in range(Epoch_Num):   
            print(f'---------------- Epoch: {epoch+1:02} ----------------')
            epoch_loss      = 0
            step            = 0 
            for Data in self.DataIn:
                # 初始化优化器的所有梯度
                self.optimizer.zero_grad()
                context,target = Data
                context     = Variable(torch.LongTensor([Vocab[context]]))   
                target      = Variable(torch.LongTensor([Vocab[target]]))
                # 判断是否送入GPU
                if self.gpucount > 0:
                    context = context.to(self.device)
                    target  = target.to(self.device)
                # 前向传播
                SkipGramVal     = self.SkipGramModel(target)
                loss            = self.criterion(SkipGramVal,context)
                epoch_loss     += loss.item()
                # 反向更新
                loss.backward()
                self.optimizer.step()
                step       += 1
                #打印输出
                if step % Print_Control == 0:
                    print('Step{} Loss:{:.3f}'.format(step,float(loss.detach().cpu().numpy())))
        print('SkipGram词嵌入训练[模型训练]已完成')
        print('EMDSize',self.SkipGramModel.embedding.weight.size())
        self.emb_save(EmbVec_path)

    # SkipGram训练完后词嵌入的存储
    def emb_save(self,EmbVec_path):
        '''
        Input
            EmbVec_path : 词嵌入保存地址 
        '''
        EmbeddedVec = self.SkipGramModel.embedding.weight.data.numpy()
        sio.savemat(EmbVec_path + 'EmbeddedVec_SkipGram.mat',{'EmbVec' : EmbeddedVec})
        print('SkipGram词嵌入训练[模型保存]已完成')


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