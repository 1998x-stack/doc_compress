import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..' + '/' + '..'))

from typing import List, Optional, Dict, Union, Tuple
import numpy as np
from collections import Counter, defaultdict
import networkx as nx
import logging

from doc_compress.graph.textrank import TextRankCompressor

class PositionRankCompressor(TextRankCompressor):
    """基于PositionRank算法的文档压缩类。
    
    该类通过继承TextRankCompressor并引入位置权重来增强文档压缩效果。
    支持带查询和不带查询两种压缩模式，可以基于最大长度或选择前N个块进行压缩。
    
    Attributes:
        doc_chunker: 文档分块器实例
        tokenize_func: 文本分词函数
        window_size: 共现窗口大小
        damping_factor: PageRank阻尼系数
        min_edge_weight: 最小边权重阈值
        position_weight_decay: 位置权重衰减因子
    """
    
    def __init__(
        self,
        doc_chunker,
        tokenize_func,
        window_size: int = 3,
        damping_factor: float = 0.85,
        min_edge_weight: float = 0.1,
        position_weight_decay: float = 0.95
    ) -> None:
        """初始化PositionRankCompressor。
        
        Args:
            doc_chunker: 文档分块器实例
            tokenize_func: 分词函数，输入文本返回token列表
            window_size: 共现窗口大小，默认为3
            damping_factor: PageRank阻尼系数，默认为0.85
            min_edge_weight: 最小边权重阈值，默认为0.1
            position_weight_decay: 位置权重衰减因子，默认为0.95
        """
        super().__init__(
            doc_chunker=doc_chunker,
            tokenize_func=tokenize_func,
            window_size=window_size,
            damping_factor=damping_factor,
            min_edge_weight=min_edge_weight
        )
        self.position_weight_decay = position_weight_decay
        self.logger = logging.getLogger(__name__)
        
    def _calculate_position_weights(
        self,
        tokens: List[str]
    ) -> Dict[str, float]:
        """计算词语的位置权重。
        
        Args:
            tokens: 分词后的token列表
            
        Returns:
            Dict[str, float]: 词语到位置权重的映射
        """
        position_weights = defaultdict(float)
        token_positions = defaultdict(list)
        
        # 记录每个词的所有出现位置
        for pos, token in enumerate(tokens, 1):
            token_positions[token].append(pos)
            
        # 计算每个词的位置权重
        for token, positions in token_positions.items():
            # 位置权重 = Σ(1/position)
            weight = sum(1.0 / pos for pos in positions)
            # 应用位置权重衰减
            weight *= self.position_weight_decay ** (min(positions) - 1)
            position_weights[token] = weight
            
        return dict(position_weights)
    
    def _calculate_chunk_position_weights(
        self,
        chunks: List[str]
    ) -> List[Dict[str, float]]:
        """计算所有文档块中词语的位置权重。
        
        Args:
            chunks: 文档块列表
            
        Returns:
            List[Dict[str, float]]: 每个块中词语的位置权重列表
        """
        chunk_position_weights = []
        
        for chunk in chunks:
            tokens = self.tokenize_func(chunk)
            weights = self._calculate_position_weights(tokens)
            chunk_position_weights.append(weights)
            
        return chunk_position_weights
        
    def _build_position_weighted_graph(
        self,
        chunks: List[str],
        query: Optional[str] = None
    ) -> Tuple[nx.Graph, List[Dict[str, float]]]:
        """构建基于位置权重的图。
        
        Args:
            chunks: 文档块列表
            query: 可选的查询文本
            
        Returns:
            Tuple[nx.Graph, List[Dict[str, float]]]: 
                位置加权图和每个块的位置权重字典列表
        """
        graph = nx.Graph()
        
        # 计算每个块中词语的位置权重
        
        chunk_position_weights = self._calculate_chunk_position_weights(chunks)
        
        # 处理查询词（如果有）
        query_tokens = set()
        if query:
            query_tokens = set(self.tokenize_func(query))
        
        # 对每个块进行分词
        chunk_tokens = [self.tokenize_func(chunk) for chunk in chunks]
        
        # 统计共现关系并构建图
        for chunk_idx, tokens in enumerate(chunk_tokens):
            position_weights = chunk_position_weights[chunk_idx]
            
            for i in range(len(tokens)):
                word1 = tokens[i]
                # 在窗口范围内查找共现词
                for j in range(i + 1, min(i + self.window_size + 1, len(tokens))):
                    word2 = tokens[j]
                    if word1 != word2:
                        # 计算边权重：位置权重乘积 * 距离衰减
                        weight = (
                            position_weights.get(word1, 0) * 
                            position_weights.get(word2, 0) * 
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
        
        return graph, chunk_position_weights
    
    def _calculate_chunk_scores(
        self,
        chunks: List[str],
        query: Optional[str] = None
    ) -> np.ndarray:
        """计算文档块的PositionRank得分。
        
        Args:
            chunks: 文档块列表
            query: 可选的查询文本
            
        Returns:
            numpy.ndarray: 每个块的得分
        """
        # 构建位置加权图
        graph, chunk_position_weights = self._build_position_weighted_graph(chunks, query)
        
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
            position_weights = chunk_position_weights[i]
            
            # 计算块得分：PageRank得分 * 位置权重
            chunk_score = sum(
                pagerank_scores.get(token, 0) * position_weights.get(token, 0)
                for token in set(tokens)
            )
            
            if query:
                # 如果有查询，增加与查询相关词的权重
                query_tokens = set(self.tokenize_func(query))
                relevance = sum(
                    2 * pagerank_scores.get(token, 0) * position_weights.get(token, 0)
                    for token in set(tokens) & query_tokens
                )
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
        """使用PositionRank算法压缩文档。
        
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
            chunks,_= self.doc_chunker.batch_chunk([doc])
            chunks=chunks[0]
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
                f"PositionRank参数: window_size={self.window_size}, "
                f"damping_factor={self.damping_factor}, "
                f"position_weight_decay={self.position_weight_decay}"
            )
            
            # 返回压缩结果
            return ''.join(selected_chunks),compression_ratio
            
        except Exception as e:
            self.logger.error(f"文档压缩过程中发生错误: {str(e)}")
            raise