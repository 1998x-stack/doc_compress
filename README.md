# doc_compress

## 关键词提取应用在文档压缩的假设
1. 根据提取的关键词，选择包含这些关键词的句子。这些句子通常包含文档的主要内容，是文档摘要的候选部分。
2. 关键词的出现位置和频率等统计学特征可以帮助判断哪些句子最能代表文档的核心信息。

https://github.com/boudinfl/pke 

- Unsupervised models
    - Statistical models
        - TfIdf
        - BM25
        - [KPMiner](http://www.aclweb.org/anthology/S10-1041.pdf)
        - [YAKE](https://doi.org/10.1016/j.ins.2019.09.013)
    - Graph-based models
        - [TextRank](http://www.aclweb.org/anthology/W04-3252.pdf)
        - [SingleRank](http://www.aclweb.org/anthology/C08-1122.pdf)
        - [TopicRank](http://aclweb.org/anthology/I13-1062.pdf)
        - [TopicalPageRank](http://users.intec.ugent.be/cdvelder/papers/2015/sterckx2015wwwb.pdf)
        - [PositionRank](http://www.aclweb.org/anthology/P17-1102.pdf)
        - [MultipartiteRank](https://arxiv.org/abs/1803.08721)

## 算法

### Statistical models
#### TFIDF
- [TFIDF文档](doc/TFIDF.md)
- [TFIDF代码](src/statistical/tfidf.py)


#### BM25
- [BM25文档](doc/BM25.md)
- [BM25代码](src/statistical/bm25.py)



## 总结

- 简单高效：TextRank、SingleRank 对短文档表现良好，适合主题单一的文本。
- 语义增强：TopicRank 和 TopicalPageRank 通过引入语义聚类和主题建模，更适合长文档和多主题场景。
- 位置敏感：PositionRank 在需要强调关键词位置时效果显著。
- 多主题覆盖：MultipartiteRank 是最适合处理多主题、多领域复杂文档的算法。



