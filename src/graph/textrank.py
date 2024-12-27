import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..' + '/' + '..'))


from typing import List, Optional, Dict, Union, Callable
import numpy as np
from collections import Counter, defaultdict
import networkx as nx
import logging
from enum import Enum
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class EdgeType(Enum):
    """Edge type for TextRank graph construction."""
    COOCCURRENCE = "cooccurrence"
    SEMANTIC = "semantic"

class TextRankCompressor:
    """TextRank-based document compression class.
    
    This class implements document compression using TextRank algorithm, supporting both
    co-occurrence and semantic similarity for edge construction.
    
    Attributes:
        doc_chunker: DocChunker instance for text chunking
        tokenize_func: Function for text tokenization
        edge_type: Type of edges to use (co-occurrence or semantic)
        window_size: Window size for co-occurrence calculation
        damping_factor: Damping factor for PageRank algorithm
        similarity_func: Custom similarity function for semantic edges
    """
    
    def __init__(
        self,
        doc_chunker,
        tokenize_func,
        edge_type: EdgeType = EdgeType.COOCCURRENCE,
        window_size: int = 3,
        damping_factor: float = 0.85,
        similarity_func: Optional[Callable] = None,
        min_edge_weight: float = 0.1
    ) -> None:
        """Initialize TextRankCompressor.
        
        Args:
            doc_chunker: Instance of DocChunker for text chunking
            tokenize_func: Function that takes text and returns token list
            edge_type: Type of edges to use (default: COOCCURRENCE)
            window_size: Window size for co-occurrence (default: 3)
            damping_factor: Damping factor for PageRank (default: 0.85)
            similarity_func: Custom similarity function (default: None)
            min_edge_weight: Minimum edge weight threshold (default: 0.1)
        """
        self.doc_chunker = doc_chunker
        self.tokenize_func = tokenize_func
        self.edge_type = edge_type
        self.window_size = window_size
        self.damping_factor = damping_factor
        self.similarity_func = similarity_func or cosine_similarity
        self.min_edge_weight = min_edge_weight
        self.logger = logging.getLogger(__name__)
        
    def _build_cooccurrence_graph(self, tokens_list: List[List[str]]) -> nx.Graph:
        """Build co-occurrence based graph from tokens.
        
        Args:
            tokens_list: List of token lists from each chunk
            
        Returns:
            networkx.Graph: Graph with co-occurrence edges
        """
        graph = nx.Graph()
        
        # 统计词共现关系
        word_pairs = defaultdict(float)
        
        for tokens in tokens_list:
            for i in range(len(tokens)):
                # 在窗口大小内统计共现
                for j in range(i + 1, min(i + self.window_size + 1, len(tokens))):
                    if tokens[i] != tokens[j]:
                        pair = tuple(sorted([tokens[i], tokens[j]]))
                        word_pairs[pair] += 1.0 / (j - i)  # 距离越远权重越小
        
        # 构建图
        for (word1, word2), weight in word_pairs.items():
            if weight >= self.min_edge_weight:
                graph.add_edge(word1, word2, weight=weight)
                
        return graph
    
    def _build_semantic_graph(self, tokens_list: List[List[str]]) -> nx.Graph:
        """Build semantic similarity based graph from tokens.
        
        Args:
            tokens_list: List of token lists from each chunk
            
        Returns:
            networkx.Graph: Graph with semantic similarity edges
        """
        graph = nx.Graph()
        
        # 将token列表转换为文本，用于向量化
        texts = [" ".join(tokens) for tokens in tokens_list]
        
        # 构建词向量
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(texts)
        
        # 计算相似度矩阵
        similarity_matrix = self.similarity_func(vectors.toarray())
        
        # 获取特征名称
        feature_names = vectorizer.get_feature_names_out()
        
        # 构建图
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                weight = similarity_matrix[i, j]
                if weight >= self.min_edge_weight:
                    graph.add_edge(feature_names[i], feature_names[j], weight=weight)
                    
        return graph
    
    def _calculate_textrank(self, chunks: List[str], query: Optional[str] = None) -> np.ndarray:
        """Calculate TextRank scores for document chunks.
        
        Args:
            chunks: List of text chunks
            query: Optional query text for guided compression
            
        Returns:
            numpy.ndarray: TextRank scores for each chunk
        """
        # 对所有块进行分词
        chunk_tokens = [self.tokenize_func(chunk) for chunk in chunks]
        
        # 根据边类型构建图
        if self.edge_type == EdgeType.COOCCURRENCE:
            graph = self._build_cooccurrence_graph(chunk_tokens)
        else:  # EdgeType.SEMANTIC
            graph = self._build_semantic_graph(chunk_tokens)
            
        # 计算PageRank
        pagerank_scores = nx.pagerank(
            graph,
            alpha=self.damping_factor,
            weight='weight'
        )
        
        # 计算每个块的得分
        scores = np.zeros(len(chunks))
        
        if query:
            # 查询模式：计算每个块与查询的相关性
            query_tokens = set(self.tokenize_func(query))
            
            for i, tokens in enumerate(chunk_tokens):
                chunk_score = 0
                for token in set(tokens):
                    if token in query_tokens:
                        chunk_score += pagerank_scores.get(token, 0)
                scores[i] = chunk_score
        else:
            # 非查询模式：使用块中词的PageRank得分总和
            for i, tokens in enumerate(chunk_tokens):
                scores[i] = sum(pagerank_scores.get(token, 0) for token in set(tokens))
                
        return scores
    
    def _get_chunks_by_length(self, chunks: List[str], scores: np.ndarray, 
                            max_length: int) -> List[str]:
        """Get chunks within max_length constraint.
        
        Args:
            chunks: List of text chunks
            scores: TextRank scores for chunks
            max_length: Maximum total length allowed
            
        Returns:
            List[str]: Selected chunks within length constraint
        """
        # 获取块长度
        chunk_lengths = [len(chunk) for chunk in chunks]
        
        # 按分数排序的索引
        sorted_indices = np.argsort(-scores)
        
        # 选择块直到达到长度限制
        selected_indices = []
        current_length = 0
        
        for idx in sorted_indices:
            if current_length + chunk_lengths[idx] <= max_length:
                selected_indices.append(idx)
                current_length += chunk_lengths[idx]
            else:
                break
                
        # 按原始顺序排序选中的块
        return [chunks[i] for i in sorted(selected_indices)]
    
    def _get_chunks_by_topn(self, chunks: List[str], scores: np.ndarray, 
                           topn: int) -> List[str]:
        """Get top N chunks by TextRank score.
        
        Args:
            chunks: List of text chunks
            scores: TextRank scores for chunks
            topn: Number of top chunks to select
            
        Returns:
            List[str]: Selected top N chunks
        """
        # 获取前N个最高分数的索引
        top_indices = np.argsort(-scores)[:topn]
        
        # 按原始顺序返回选中的块
        return [chunks[i] for i in sorted(top_indices)]
    
    def compress(
        self,
        doc: str,
        max_length: Optional[int] = None,
        topn: Optional[int] = None,
        query: Optional[str] = None,
        edge_type: Optional[EdgeType] = None,
        window_size: Optional[int] = None,
        damping_factor: Optional[float] = None
    ) -> str:
        """Compress document using TextRank scoring.
        
        Args:
            doc: Input document text
            max_length: Maximum length of compressed document
            topn: Number of top chunks to select
            query: Optional query text for guided compression
            edge_type: Optional override for edge type
            window_size: Optional override for window size
            damping_factor: Optional override for damping factor
            
        Returns:
            str: Compressed document text
            
        Raises:
            ValueError: If neither max_length nor topn is specified,
                      or if input parameters are invalid
        """
        # 参数验证
        if not doc:
            raise ValueError("Empty document provided")
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive")
        if topn is not None and topn <= 0:
            raise ValueError("topn must be positive")
        if max_length is None and topn is None:
            raise ValueError("Either max_length or topn must be specified")
            
        # 临时更新参数（如果提供）
        original_params = {
            'edge_type': self.edge_type,
            'window_size': self.window_size,
            'damping_factor': self.damping_factor
        }
        
        if edge_type is not None:
            self.edge_type = edge_type
        if window_size is not None:
            self.window_size = window_size
        if damping_factor is not None:
            self.damping_factor = damping_factor
            
        try:
            # 文档分块
            chunks = self.doc_chunker.batch_chunk([doc])[0]
            if not chunks:
                self.logger.warning("Document chunking produced no chunks")
                return ""
                
            # 计算TextRank分数
            scores = self._calculate_textrank(chunks, query)
            
            # 根据条件选择压缩方式
            if max_length is not None:
                selected_chunks = self._get_chunks_by_length(chunks, scores, max_length)
            else:
                # 如果指定了topN，使用topN方式
                topn = min(topn, len(chunks))  # 确保topN不超过块数
                selected_chunks = self._get_chunks_by_topn(chunks, scores, topn)
                
            # 记录压缩信息
            compression_ratio = len(''.join(selected_chunks)) / len(doc)
            self.logger.info(f"Compression ratio: {compression_ratio:.2%}")
            self.logger.info(
                f"TextRank parameters: edge_type={self.edge_type.value}, "
                f"window_size={self.window_size}, damping_factor={self.damping_factor}"
            )
            
            return ''.join(selected_chunks)
            
        finally:
            # 恢复原始参数
            self.__dict__.update(original_params)