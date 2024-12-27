import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..' + '/' + '..'))


from typing import List, Optional, Dict
import numpy as np
import networkx as nx
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from doc_compress.graph.textrank import TextRankCompressor

class SingleRankCompressor(TextRankCompressor):
    """基于SingleRank算法的文档压缩类。
    
    该类通过继承TextRankCompressor并引入TF-IDF权重来增强文档压缩效果。
    支持带查询和不带查询两种压缩模式，可以基于最大长度或选择前N个块进行压缩。
    
    Attributes:
        doc_chunker: 文档分块器实例
        tokenize_func: 文本分词函数
        window_size: 共现窗口大小
        damping_factor: PageRank阻尼系数
        min_edge_weight: 最小边权重阈值
        tfidf_vectorizer: TF-IDF向量化器
    """
    
    def __init__(
        self,
        doc_chunker,
        tokenize_func,
        window_size: int = 3,
        damping_factor: float = 0.85,
        min_edge_weight: float = 0.1
    ) -> None:
        """初始化SingleRankCompressor。
        
        Args:
            doc_chunker: 文档分块器实例
            tokenize_func: 分词函数，输入文本返回token列表
            window_size: 共现窗口大小，默认为3
            damping_factor: PageRank阻尼系数，默认为0.85
            min_edge_weight: 最小边权重阈值，默认为0.1
        """
        super().__init__(
            doc_chunker=doc_chunker,
            tokenize_func=tokenize_func,
            window_size=window_size,
            damping_factor=damping_factor,
            min_edge_weight=min_edge_weight
        )
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=tokenize_func,
            lowercase=False  # 保持原始大小写
        )
        self.logger = logging.getLogger(__name__)
    
    def _calculate_tfidf_weights(self, chunks: List[str]) -> Dict[str, float]:
        """计算文档块中词语的TF-IDF权重。
        
        Args:
            chunks: 文档块列表
            
        Returns:
            Dict[str, float]: 词语到TF-IDF权重的映射
        """
        # 计算TF-IDF矩阵
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunks)
        
        # 获取特征名称（词语）
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # 计算每个词的平均TF-IDF权重
        avg_weights = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # 创建词语到权重的映射
        return dict(zip(feature_names, avg_weights))
    
    def _build_weighted_graph(
        self, 
        chunks: List[str],
        query: Optional[str] = None
    ) -> nx.Graph:
        """构建加权图。
        
        使用TF-IDF权重和共现关系构建图，
        如果提供查询，则增加与查询相关词语的权重。
        
        Args:
            chunks: 文档块列表
            query: 可选的查询文本
            
        Returns:
            networkx.Graph: 加权图
        """
        graph = nx.Graph()
        
        # 计算TF-IDF权重
        tfidf_weights = self._calculate_tfidf_weights(chunks)
        
        # 处理查询词（如果有）
        query_tokens = set()
        if query:
            query_tokens = set(self.tokenize_func(query))
        
        # 对每个块进行分词
        chunk_tokens = [self.tokenize_func(chunk) for chunk in chunks]
        
        # 统计共现关系并构建图
        for tokens in chunk_tokens:
            for i in range(len(tokens)):
                word1 = tokens[i]
                # 在窗口范围内查找共现词
                for j in range(i + 1, min(i + self.window_size + 1, len(tokens))):
                    word2 = tokens[j]
                    if word1 != word2:
                        # 计算边权重：TF-IDF权重乘积 * 距离衰减
                        weight = (
                            tfidf_weights.get(word1, 0) * 
                            tfidf_weights.get(word2, 0) * 
                            (1.0 / (j - i))
                        )
                        
                        # 如果是查询相关词，增加权重
                        if query and (word1 in query_tokens or word2 in query_tokens):
                            weight *= 2.0
                            
                        if weight >= self.min_edge_weight:
                            if graph.has_edge(word1, word2):
                                graph[word1][word2]['weight'] += weight
                            else:
                                graph.add_edge(word1, word2, weight=weight)
        
        return graph
    
    def _calculate_chunk_scores(
        self,
        chunks: List[str],
        query: Optional[str] = None
    ) -> np.ndarray:
        """计算文档块的SingleRank得分。
        
        Args:
            chunks: 文档块列表
            query: 可选的查询文本
            
        Returns:
            numpy.ndarray: 每个块的得分
        """
        # 构建加权图
        graph = self._build_weighted_graph(chunks, query)
        
        # 如果图为空，返回零分数组
        if not graph:
            self.logger.warning("Empty graph constructed")
            return np.zeros(len(chunks))
        
        # 计算PageRank得分
        pagerank_scores = nx.pagerank(
            graph,
            alpha=self.damping_factor,
            weight='weight'
        )
        
        # 计算每个块的得分
        chunk_tokens = [self.tokenize_func(chunk) for chunk in chunks]
        scores = np.zeros(len(chunks))
        
        for i, tokens in enumerate(chunk_tokens):
            # 累加块中每个词的PageRank得分
            chunk_score = sum(pagerank_scores.get(token, 0) for token in set(tokens))
            
            if query:
                # 如果有查询，增加与查询相关词的权重
                query_tokens = set(self.tokenize_func(query))
                relevance = sum(2 * pagerank_scores.get(token, 0) 
                              for token in set(tokens) & query_tokens)
                chunk_score += relevance
                
            scores[i] = chunk_score
            
        return scores
    
    def compress(
        self,
        doc: str,
        query: Optional[str] = None,
        max_length: Optional[int] = None,
        topn: Optional[int] = None
    ) -> str:
        """使用SingleRank算法压缩文档。
        
        Args:
            doc: 输入文档文本
            query: 可选的查询文本
            max_length: 压缩后文档的最大长度
            topn: 选择得分最高的前N个块
            
        Returns:
            str: 压缩后的文档文本
            
        Raises:
            ValueError: 当参数无效或文档为空时
        """
        # 参数验证
        if not doc:
            raise ValueError("文档为空")
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length必须为正数")
        if topn is not None and topn <= 0:
            raise ValueError("topn必须为正数")
        if max_length is None and topn is None:
            raise ValueError("必须指定max_length或topn其中之一")
            
        try:
            # 文档分块
            chunks = self.doc_chunker.batch_chunk([doc])[0]
            if not chunks:
                self.logger.warning("文档分块结果为空")
                return ""
            
            # 计算块得分
            scores = self._calculate_chunk_scores(chunks, query)
            
            # 根据条件选择压缩方式
            if max_length is not None:
                selected_chunks = self._get_chunks_by_length(chunks, scores, max_length)
            else:
                topn = min(topn, len(chunks))  # 确保topN不超过块数
                selected_chunks = self._get_chunks_by_topn(chunks, scores, topn)
            
            # 记录压缩信息
            compression_ratio = len(''.join(selected_chunks)) / len(doc)
            self.logger.info(f"压缩比率: {compression_ratio:.2%}")
            self.logger.info(
                f"SingleRank参数: window_size={self.window_size}, "
                f"damping_factor={self.damping_factor}"
            )
            
            # 返回压缩结果
            return ''.join(selected_chunks)
            
        except Exception as e:
            self.logger.error(f"文档压缩过程中发生错误: {str(e)}")
            raise