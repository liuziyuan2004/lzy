# 书写项目文档

## 一、多项式朴素贝叶斯分类器实现

### 1.1 核心概率模型

**条件独立性假设**：
- 假设所有特征（词项）在给定类别条件下相互独立
- 数学表达：
  ![TF-IDF运行结果](path/to/tfidf_feat.png)
  $$
  P(\mathbf{x}|y) = \prod_{i=1}^n P(x_i|y)
  $$
  

**多项式分布假设**：
- 特征服从多项式分布，建模词项出现次数：
![TF-IDF运行结果](path/to/tfidf_feat.png)
  $$
  P(\mathbf{x}|y) = \frac{(\sum_i x_i)!}{\prod_i x_i!}\prod_i P(w_i|y)^{x_i}
  $$

### 1.2 贝叶斯分类决策

**分类决策函数**：
$$
\hat{y} = \arg\max_{y} P(y)\prod_{i=1}^n P(x_i|y)
$$
![TF-IDF运行结果](path/to/tfidf_feat.png)
**具体实现步骤**：

1. 先验概率计算：
   $$
   P(y) = \frac{N_y + \alpha}{N + \alpha K}
   $$
   其中$N_y$是类别$y$的样本数，$K$是类别数，$\alpha$为平滑系数
![TF-IDF运行结果](path/to/tfidf_feat.png)
2. 条件概率计算（使用拉普拉斯平滑）：
   $$
   P(w_i|y) = \frac{count(w_i,y) + \alpha}{\sum_w count(w,y) + \alpha|V|}
   $$
   $|V|$为词汇表大小

3. 对数空间计算（避免下溢）：
   $$
   \log P(y|\mathbf{x}) \propto \log P(y) + \sum_{i=1}^n x_i \log P(w_i|y)
   $$
![TF-IDF运行结果](path/to/tfidf_feat.png)
### 1.3 算法优势分析

| 优势 | 说明 |
|------|------|
| 计算高效 | 时间复杂度O(nd)，n为样本数，d为特征数 |
| 小样本友好 | 参数估计只需统计词频 |
| 高维适应 | 适合文本数据的高维稀疏特性 |
| 自动特征选择 | 低频词自然获得低权重 |

## 二、特征工程实现对比

### 2.1 高频词特征选择

**数学表达**：
$$
\text{Features} = \underset{t\in V}{\text{topk}} \left( \sum_{d\in D} \mathbb{I}(t\in d) \right)
$$
![TF-IDF运行结果](path/to/tfidf_feat.png)
**实现代码**：
```python
vectorizer = CountVectorizer(
    max_features=1000,
    tokenizer=jieba.cut,
    stop_words=stopwords
)
```

**特点分析**：
- 计算复杂度：O(NL)，N为文档数，L为平均文档长度
- 内存消耗：存储词频向量（稀疏矩阵）
- 局限性：可能保留高频但无区分度的词

### 2.2 TF-IDF特征加权

**数学定义**：
$$
\text{tfidf}(t,d,D) = \underbrace{\frac{f_{t,d}}{\sum_{t'}f_{t',d}}}_{\text{tf}} \times \underbrace{\log\frac{|D|}{|\{d'\in D:t\in d'\}|}}_{\text{idf}}
$$
![TF-IDF运行结果](path/to/tfidf_feat.png)
**实现代码**：
```python
vectorizer = TfidfVectorizer(
    max_features=1000,
    tokenizer=jieba.cut,
    stop_words=stopwords,
    norm='l2'
)
```

**特性对比**：

| 特性 | 高频词 | TF-IDF |
|------|--------|--------|
| 权重类型 | 整数频次 | 连续值 |
| 常见词处理 | 保留 | 抑制 |
| 稀有词处理 | 忽略 | 增强 |
| 计算开销 | 低 | 中等 |
| 效果 | 基线 | 通常更优 |

## 三、系统运行结果

### 3.1 高频词特征分类效果
![高频词特征运行结果](path/to/freq_feat.png)

### 3.2 TF-IDF特征分类效果  
![TF-IDF运行结果](path/to/tfidf_feat.png)


