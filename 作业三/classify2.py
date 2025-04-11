from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureSelector:
    def __init__(self, feature_type='frequency', max_features=None,
                 ngram_range=(1, 1), stop_words=None, vocabulary=None):
        """
        参数化特征选择器

        参数:
        - feature_type: str, 可选 'frequency' 或 'tfidf'，默认 'frequency'
        - max_features: int or None, 选择最大特征数
        - ngram_range: tuple, n-gram范围，如(1,1)表示仅使用unigram
        - stop_words: str or list, 停用词
        - vocabulary: 自定义词汇表
        """
        self.feature_type = feature_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.vocabulary = vocabulary
        self.vectorizer = None
        self._initialize_vectorizer()

    def _initialize_vectorizer(self):
        """根据feature_type初始化适当的向量化器"""
        if self.feature_type == 'frequency':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words=self.stop_words,
                vocabulary=self.vocabulary
            )
        elif self.feature_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words=self.stop_words,
                vocabulary=self.vocabulary
            )
        else:
            raise ValueError(f"不支持的feature_type: {self.feature_type}. 请选择 'frequency' 或 'tfidf'")

    def fit(self, raw_documents):
        """拟合向量化器"""
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, raw_documents):
        """将文本转换为特征矩阵"""
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, raw_documents):
        """拟合并转换文本数据"""
        return self.vectorizer.fit_transform(raw_documents)

    def get_feature_names(self):
        """获取特征名称"""
        return self.vectorizer.get_feature_names_out()

    def set_params(self, **params):
        """设置参数并重新初始化向量化器"""
        for key, value in params.items():
            setattr(self, key, value)
        self._initialize_vectorizer()
        return self


# ============ 使用示例 ============

# 1. 准备示例数据
corpus = [
    '这是第一个文档。',
    '这个文档是第二个文档。',
    '而这是第三个文档。',
    '这第一个文档吗?',
]

print("原始语料库:")
for i, doc in enumerate(corpus):
    print(f"文档{i + 1}: {doc}")

# 2. 使用高频词特征
print("\n=== 高频词特征 ===")
fs_freq = FeatureSelector(feature_type='frequency', max_features=5)
X_freq = fs_freq.fit_transform(corpus)

print("\n特征名称:")
print(fs_freq.get_feature_names())

print("\n特征矩阵:")
print(X_freq.toarray())

# 3. 使用TF-IDF特征
print("\n=== TF-IDF特征 ===")
fs_tfidf = FeatureSelector(feature_type='tfidf', max_features=5)
X_tfidf = fs_tfidf.fit_transform(corpus)

print("\n特征名称:")
print(fs_tfidf.get_feature_names())

print("\n特征矩阵:")
print(X_tfidf.toarray())

# 4. 动态切换参数示例
print("\n=== 动态切换参数示例 ===")
fs = FeatureSelector(feature_type='frequency', max_features=3)
print("初始特征名称 (frequency, max_features=3):")
print(fs.fit_transform(corpus).toarray())
print(fs.get_feature_names())

# 切换到TF-IDF并增加max_features
fs.set_params(feature_type='tfidf', max_features=5)
print("\n切换后特征名称 (tfidf, max_features=5):")
print(fs.fit_transform(corpus).toarray())
print(fs.get_feature_names())
