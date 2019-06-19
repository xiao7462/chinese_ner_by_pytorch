# chinese_ner_by_pytorch
## 根据pytorch [ner教程](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html) 实现的中文ner

# 使用方法
linux下直接输入  `nohup /data1/tangx/anaconda3/bin/python3 -u blstm+crf.py > log.out 2>&1 &`     
运行
![pgn](https://i.loli.net/2019/06/19/5d09957f5aebd19259.png)

# 预测
![pgn](https://i.loli.net/2019/06/19/5d0995841059869007.png)

# 改进
可以加入 minibatch 和bert/NRNIE的预训练      

人民日报训练集结果，参考自他人的结果
![pgn](https://i.loli.net/2019/06/19/5d099616d6a0370024.png)
