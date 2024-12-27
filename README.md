# doc_compress

## 关键词提取应用在文档压缩的假设
1. 根据提取的关键词，选择包含这些关键词的句子。这些句子通常包含文档的主要内容，是文档摘要的候选部分。
2. 关键词的出现位置和频率等统计学特征可以帮助判断哪些句子最能代表文档的核心信息。

https://github.com/boudinfl/pke 

- Unsupervised models
    - Statistical models
        - FirstPhrases
        - TfIdf
        - KPMiner http://www.aclweb.org/anthology/S10-1041.pdf 
        - YAKE https://doi.org/10.1016/j.ins.2019.09.013 
    - Graph-based models
        - TextRank http://www.aclweb.org/anthology/W04-3252.pdf 
        - SingleRank http://www.aclweb.org/anthology/C08-1122.pdf 
        - TopicRank http://aclweb.org/anthology/I13-1062.pdf 
        - TopicalPageRank http://users.intec.ugent.be/cdvelder/papers/2015/sterckx2015wwwb.pdf 
        - PositionRank http://www.aclweb.org/anthology/P17-1102.pdf 
        - MultipartiteRank https://arxiv.org/abs/1803.08721 

## 算法

### TFIDF
doc/TFIDF.md


