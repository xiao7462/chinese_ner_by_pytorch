#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-25 11:16:44
# @Author  : Xing Tang (xiao7462@gmail.com)
# @Link    : https://github.com/xiao7462
# @Version : $Id$
# Author: Tang xing
# 代码来自pytorch 官方教程
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1) # 返回每一行最大值，以及它的索引
    return idx.item()


def prepare_sequence(seq, to_ix): # 返回单词的索引
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec): #vec是1*5, type是Variable
    max_score = vec[0, argmax(vec)]
    #max_score维度是１，　max_score.view(1,-1)维度是１＊１，max_score.view(1, -1).expand(1, vec.size()[1])的维度是１＊５
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1]) # view(1, -1)将向量转换为行向量   # vec.size()维度是1*5
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast))) #先做减法的原因在于，减去最大值可以避免e的指数次，计算机上溢。等同于return torch.log(torch.sum(torch.exp(vec)))
        

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)# self.tagset_size = 5

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))  #说明了转移矩阵是随机的，并且会更新
        # transitions 为(5,5)的向量 ，也就是CRF层的转移矩阵A
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden() # hidden 层的大小

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))
    # 预测序列的得分
    def _forward_alg(self, feats): #这个函数，只是根据 随机的transitions ，前向传播算出的一个score，用到了动态规划的思想，但是因为用的是随机的转移矩阵，算出的值很大 score>20
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # init_alphas = tensor([[-10000., -10000., -10000., -10000., -10000.]])  1*5
        
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        #因为start tag是4，所以tensor([[-10000., -10000., -10000.,      0., -10000.]])，将start的值为零，表示开始进行网络的传播，

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        #初始状态的forward_var，随着step t变化
        # forward_var = tensor([[-10000., -10000., -10000.,      0., -10000.]]) 1*5
        
        # Iterate through the sentence
        for feat in feats: #feat的维度是５ 依次把每一行取出来
        #feats = self._get_lstm_features(sentence)  feats的维度是 (len(sentences), len(tag_to_ix))
        # 因此 feat就是对每一个单词进行遍历，每一个单词有5个tag的维度
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size): #next tag 就是简单 i，从0到len(tag_to_ix) 取值 0-5
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size) #维度是1*5 ,LSTM后的矩阵被当成了 emit score
                     # view(1,-1） 将tag的分数转换为行向量，expand(1,tagset_size) 转换为(1,5)的行向量
                    
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                #维度是１＊５
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                #第一次迭代时理解：
                # trans_score所有其他标签到Ｂ标签的概率
                # 由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊５，5个值是相同的
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
                #此时的alphas t 是一个长度为5，例如<class 'list'>: [tensor(0.8259), tensor(2.1739), tensor(1.3526), tensor(-9999.7168), tensor(-0.7102)]
            forward_var = torch.cat(alphas_t).view(1, -1)
            # forward_var = tensor([[ 0.2830,  2.0379,  0.1633, -0.0177,  0.8699]]) 举例
            #到第（t-1）step时５个标签的各自分数
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence): #函数里经过了embedding，lstm，linear层，是根据LSTM算出的一个矩阵。这里是11x5的一个tensor，而这个11x5的tensor，就是发射矩阵

    #Bi-LSTM layer的输出维度是tag size ,相当于每个词wi映射到tag的发射概率，设Bi-LSTM的输出矩阵为P，其中Pij代表词wi映射到tagj的非归一化概率

        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out) # 将hidden通过线性层转换为taget
        return lstm_feats

    def _score_sentence(self, feats, tags):
        '''
        是根据真实的标签算出的一个score，这与上面的def _forward_alg(self, feats)有什么不同的地方嘛？共同之处在于，两者都是用的随机的转移矩阵算的score，但是不同地方在于，上面那个函数算了一个最大可能路径，但是实际上可能不是真实的 各个标签转移的值。例如说，真实的标签 是 N V V，但是因为transitions是随机的，所以上面的函数得到的其实是N  N N这样，两者之间的score就有了差距。而后来的反向传播，就能够更新transitions，使得转移矩阵逼近真实的“转移矩阵”。
        '''
        # Gives the score of a provided tag sequence #feats 11*5  tag 11 维
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])#将START_TAG的标签３拼接到tag序列最前面，这样tag就是12个了
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        # 维特比解码， 实际上就是在预测的时候使用了， 输出得分与路径值
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0 #  将start设置为0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # 其他标签（B,I,E,Start,End）到标签next_tag的概率
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # 从step0到step(i-1)时5个序列中每个序列的最大score
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # transition to STOP_TAG
        # 其他标签到STOP_TAG的转移概率
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse() # 把从后向前的路径正过来
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence) #11*5 经过了LSTM+Linear矩阵后的输出，之后作为CRF的输入。
        forward_score = self._forward_alg(feats)#0维的一个得分，20.*来着
        gold_score = self._score_sentence(feats, tags)#tensor([ 4.5836])
        return forward_score - gold_score #这是两者之间的差值，后来直接根据这个差值，反向传播 分别用预测的得分-真实的得分

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
        

def read_data(path):
    f = open( path , 'r', encoding = 'utf-8')
    sentences =[]
    labels = []
    seq_data=[]
    seq_label=[]
    for i in f.readlines():
        i=i.replace("\n", "")
        lst=i.split(" ")
        
        if len(lst)==2:
            seq_data.append(lst[0])
            seq_label.append(lst[1])
            
        else: #语料中是空行分隔句子
            sent=" ".join(seq_data)
            seq_data.clear()
            sentences.append(sent)
            
            label=" ".join(seq_label)
            seq_label.clear()
            labels.append(label)   
    print (len(sentences),len(labels))
    print (sentences[0:5], labels[0:5])

    data=[]

    for i in range(len(sentences)):
        data.append( (sentences[i].split(), labels[i].split()) )

    return data

def train_data(training_data, word_to_ix,tag_to_ix,EMBEDDING_DIM = 220, HIDDEN_DIM = 128,EPOCHS=4 , lr = 0.001):
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    time_start = time.time()
    for epoch in range(EPOCHS):
        print ("epoch %d =============" % epoch)
        count = 1
        for sentence, tags in training_data:
            # 每次循环进行梯度清零
            model.zero_grad()
            # 转换句子为索引
            sentence_in = prepare_sequence(sentence, word_to_ix) 
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            # 前向传播+ 误差
            loss = model.neg_log_likelihood( sentence_in ,targets)
            # 计算误差，梯度，更新参数
            loss.backward()
            optimizer.step()
            
            if count% 500 == 0:
                print("iter %d: loss %f" %(count,loss))
            count+=1
    time_end=time.time()
    print("time used: %d s" % (time_end-time_start)  )
    return model

def compare(y, y_pred):
    error_index = []
    if len(y) == len(y_pred):
        for i in range(0, len(y)):
            if y[i] != y_pred[i]:
                error_index.append(i)

    print("error_index:",error_index)

if __name__ == '__main__':
    # 加载数据
    path = 'ner_data.txt'
    data = read_data(path)
    print (data[1])

    # 设定tag
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tag_to_ix = {}
    word_to_ix = {"<UNK>":0} #生字给id=0

    for sentence, tags in data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
                
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag]=len(tag_to_ix)
    tag_to_ix[START_TAG]=7
    tag_to_ix[STOP_TAG]=8        
    
    print (tag_to_ix)
    print (len(word_to_ix),len(tag_to_ix))

    training_data,test_data=train_test_split(data,test_size=0.2, random_state=0)

    print (len(training_data),len(test_data))
    # 训练
    model = train_data(training_data, word_to_ix,tag_to_ix)
    # 测试
    # Check predictions before training
    with torch.no_grad():  
        
        for pair in test_data[0:10]: #抽取部分看看效果
            sentence=pair[0]
            tag=pair[1]
            
            precheck_sent = prepare_sequence(sentence, word_to_ix)
            precheck_tags = torch.tensor([tag_to_ix[t] for t in tag], dtype=torch.long) 
            score,y_pred=model(precheck_sent)
                
            print(sentence)
            print(tag)
            print("实际tagid:",list(precheck_tags.numpy()))
            print("预测tagid:",y_pred)
            compare(precheck_tags,y_pred)
            
            print("====================")

    # 预测
    with torch.no_grad():
        precheck_sent = prepare_sequence("诺贝尔生理学或医学奖得主屠00：“一旦疟原虫对青0素联合疗法产生抗药性，疟疾将无药可治，人类势必遭遇一场浩劫。”中国中医科学院青0素研究中心研究员王继刚：“如果没有青0素，每年会有几百万人死亡。”别眨眼重磅新闻明天发布敬请期待！",word_to_ix)
        score,y_pred=model(precheck_sent)
        print("预测tagid:",y_pred)

    with torch.no_grad():  
        precheck_sent = prepare_sequence("我爱北京天安门", word_to_ix)
        score,y_pred=model(precheck_sent)
        print("预测tagid:",y_pred)
