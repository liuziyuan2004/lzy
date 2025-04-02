import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba.cut()方法对文本切词处理
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words

all_words = []
all_texts = []  # 用于TF-IDF的原始文本

def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    # 遍历邮件建立词库
    for filename in filename_list:
        words = get_words(filename)
        all_words.append(words)
        # 将分词后的结果用空格连接，作为原始文本
        all_texts.append(' '.join(words))
    
    # itertools.chain()把all_words内的所有列表组合成一个列表
    # collections.Counter()统计词个数
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]

def extract_features(method='frequency', top_num=100):
    """
    特征提取方法，可选择高频词特征或TF-IDF加权特征
    :param method: 'frequency' 或 'tfidf'
    :param top_num: 当method='frequency'时，选择的高频词数量
    :return: 特征向量
    """
    if method == 'frequency':
        # 高频词特征提取
        top_words = get_top_words(top_num)
        vector = []
        for words in all_words:
            word_map = list(map(lambda word: words.count(word), top_words))
            vector.append(word_map)
        return np.array(vector), top_words
    elif method == 'tfidf':
        # TF-IDF特征提取
        tfidf_vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
        return tfidf_matrix.toarray(), tfidf_vectorizer.get_feature_names_out()
    else:
        raise ValueError("method参数必须是'frequency'或'tfidf'")

# 使用示例：选择特征提取方法
method = 'frequency'  # 可以改为'tfidf'使用TF-IDF特征
vector, feature_words = extract_features(method=method)

# 0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0
labels = np.array([1]*127 + [0]*24)
model = MultinomialNB()
model.fit(vector, labels)

def predict(filename, method='frequency', feature_words=None):
    """对未知邮件分类"""
    # 获取邮件内容
    words = get_words(filename)
    text = ' '.join(words)
    
    # 构建特征向量
    if method == 'frequency':
        current_vector = np.array(
            tuple(map(lambda word: words.count(word), feature_words)))
    elif method == 'tfidf':
        # 需要使用之前创建的TF-IDF向量化器，这里简化为示例
        # 实际应用中需要保存向量化器
        tfidf_vectorizer = TfidfVectorizer(max_features=100, vocabulary=feature_words)
        current_vector = tfidf_vectorizer.fit_transform([text]).toarray()[0]
    else:
        raise ValueError("method参数必须是'frequency'或'tfidf'")
    
    # 预测结果
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'

# 测试
print('151.txt分类情况:{}'.format(predict('邮件_files/151.txt', method, feature_words)))
print('152.txt分类情况:{}'.format(predict('邮件_files/152.txt', method, feature_words)))
print('153.txt分类情况:{}'.format(predict('邮件_files/153.txt', method, feature_words)))
print('154.txt分类情况:{}'.format(predict('邮件_files/154.txt', method, feature_words)))
print('155.txt分类情况:{}'.format(predict('邮件_files/155.txt', method, feature_words)))
