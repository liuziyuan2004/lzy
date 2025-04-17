#词向量
import pandas as pd
import jieba
from gensim.models.word2vec import Word2Vec

# 读入训练集文件
data = pd.read_csv('train.csv')
# 转字符串数组
corpus = data['comment'].values.astype(str)
# 分词，再重组为字符串数组
corpus = [jieba.lcut(corpus[index]
                          .replace("，", "")
                          .replace("!", "")
                          .replace("！", "")
                          .replace("。", "")
                          .replace("~", "")
                          .replace("；", "")
                          .replace("？", "")
                          .replace("?", "")
                          .replace("【", "")
                          .replace("】", "")
                          .replace("#", "")
                        ) for index in range(len(corpus))]