import torch

Vocab_Corpus = {}
pretrained_Vec = []
Vec_index = 0
with open('/Users/xulvxiaowei/Downloads/glove.6B/glove.6B.100d.txt', 'r', encoding='utf-8', errors='ignore') as file_h:
    for line in file_h:
        try:
            Vocab_Corpus[line.split(' ')[0]] = Vec_index 
            tmpVec     = [float(i) for i in line.split(' ')[1:]]
            pretrained_Vec.append(tmpVec)
            Vec_index += 1
        except:
            print('Word2Vec[读取预训练Glove词向量]出错')

GloveVec = torch.Tensor(pretrained_Vec)
print(GloveVec.size())
