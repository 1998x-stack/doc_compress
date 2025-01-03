import sys
import os
import logging
from typing import List, Optional, Dict, Union, Any
from enum import Enum
import numpy as np
from collections import defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass


class TopicType(Enum):
    """主题类型枚举类"""

    SEMANTIC = "semantic"  # 语义相似度
    TFIDF = "tfidf"  # TF-IDF相似度


@dataclass
class CompressionResult:
    """压缩结果数据类"""

    compressed_text: str  # 压缩后的文本
    compression_ratio: float  # 压缩比例
    selected_chunks: List[str]  # 选中的文本块
    topic_scores: np.ndarray  # 主题得分


class TopicRankCompressor:
    """基于主题排序的文档压缩器

    使用主题排序算法对文档进行压缩，支持基于查询的定向压缩和无查询的通用压缩。
    主要功能包括文本分块、主题建模、图排序和文档压缩。

    Attributes:
        doc_chunker: 文档分块器实例
        tokenize_func: 分词函数
        topic_type: 主题建模类型
        window_size: 共现窗口大小
        damping_factor: PageRank阻尼系数
        min_similarity: 最小相似度阈值
        logger: 日志记录器
    """

    def __init__(
        self,
        doc_chunker: Any,
        tokenize_func: Any,
        topic_type: TopicType = TopicType.TFIDF,
        window_size: int = 3,
        damping_factor: float = 0.85,
        min_similarity: float = 0.1,
    ) -> None:
        """初始化TopicRankCompressor

        Args:
            doc_chunker: 文档分块器实例
            tokenize_func: 分词函数
            topic_type: 主题建模类型 (默认: TFIDF)
            window_size: 共现窗口大小 (默认: 3)
            damping_factor: PageRank阻尼系数 (默认: 0.85)
            min_similarity: 最小相似度阈值 (默认: 0.1)
        """
        self.doc_chunker = doc_chunker
        self.tokenize_func = tokenize_func
        self.topic_type = topic_type
        self.window_size = window_size
        self.damping_factor = damping_factor
        self.min_similarity = min_similarity

        # 配置日志记录器
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _build_topic_graph(
        self, chunks: List[str], query: Optional[str] = None
    ) -> nx.Graph:
        """构建主题图

        基于文本块构建主题相似度图，支持查询导向的图构建

        Args:
            chunks: 文本块列表
            query: 可选的查询文本

        Returns:
            networkx.Graph: 主题图
        """
        # 初始化图
        graph = nx.Graph()

        # 文本向量化
        vectorizer = TfidfVectorizer(tokenizer=self.tokenize_func, lowercase=True)

        # 如果有查询，将查询添加到文本块中
        if query:
            all_texts = chunks + [query]
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            query_vector = tfidf_matrix[-1]
            chunk_vectors = tfidf_matrix[:-1]
        else:
            tfidf_matrix = vectorizer.fit_transform(chunks)
            chunk_vectors = tfidf_matrix

        # 计算块间相似度
        similarity_matrix = cosine_similarity(chunk_vectors)

        # 构建图边
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                weight = similarity_matrix[i, j]
                if weight >= self.min_similarity:
                    graph.add_edge(i, j, weight=weight)

        return graph

    def _calculate_topic_scores(
        self, chunks: List[str], query: Optional[str] = None
    ) -> np.ndarray:
        """计算主题得分

        使用PageRank算法计算每个主题的重要性得分

        Args:
            chunks: 文本块列表
            query: 可选的查询文本

        Returns:
            np.ndarray: 主题得分数组
        """
        # 构建主题图
        graph = self._build_topic_graph(chunks, query)

        # 如果图为空，返回均匀分布的得分
        if not graph.nodes():
            self.logger.warning("Empty topic graph, returning uniform scores")
            return np.ones(len(chunks)) / len(chunks)

        # 计算PageRank得分
        pagerank_scores = nx.pagerank(graph, alpha=self.damping_factor, weight="weight")

        # 初始化得分数组
        scores = np.zeros(len(chunks))
        for i in range(len(chunks)):
            scores[i] = pagerank_scores.get(i, 0)

        # 如果有查询，调整得分
        if query:
            query_tokens = set(self.tokenize_func(query))
            for i, chunk in enumerate(chunks):
                chunk_tokens = set(self.tokenize_func(chunk))
                query_overlap = len(chunk_tokens & query_tokens)
                if query_overlap > 0:
                    scores[i] *= 1 + query_overlap / len(query_tokens)

        return scores

    def _select_chunks(
        self,
        chunks: List[str],
        scores: np.ndarray,
        max_length: Optional[int] = None,
        topn: Optional[int] = None,
    ) -> List[str]:
        """选择文本块

        基于得分和约束条件选择文本块

        Args:
            chunks: 文本块列表
            scores: 主题得分
            max_length: 最大长度约束
            topn: 选择前N个块

        Returns:
            List[str]: 选中的文本块列表
        """
        # 获取块的原始索引和长度
        indices = list(range(len(chunks)))
        lengths = [len(chunk) for chunk in chunks]

        # 根据得分对索引排序
        sorted_pairs = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        sorted_indices = [pair[0] for pair in sorted_pairs]

        selected_indices = []

        if max_length is not None:
            # 基于最大长度选择
            current_length = 0
            for idx in sorted_indices:
                if current_length + lengths[idx] <= max_length:
                    selected_indices.append(idx)
                    current_length += lengths[idx]
                else:
                    break
        else:
            # 选择前N个块
            selected_indices = sorted_indices[:topn]

        # 按原始顺序返回选中的块
        return [chunks[i] for i in sorted(selected_indices)]

    def compress(
        self,
        doc: str,
        query: Optional[str] = None,
        max_length: Optional[int] = None,
        topn: Optional[int] = None,
    ) -> CompressionResult:
        """压缩文档

        使用主题排序方法压缩文档，支持查询导向的压缩

        Args:
            doc: 输入文档
            query: 可选的查询文本
            max_length: 压缩后的最大长度
            topn: 选择前N个块

        Returns:
            CompressionResult: 压缩结果对象

        Raises:
            ValueError: 参数无效或文档为空时抛出
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

        try:
            # 文档分块
            chunks, _ = self.doc_chunker.batch_chunk([doc])
            chunks = chunks[0]
            if not chunks:
                self.logger.warning("Document chunking produced no chunks")
                return CompressionResult("", 0.0, [], np.array([]))

            # 计算主题得分
            scores = self._calculate_topic_scores(chunks, query)

            # 选择文本块
            selected_chunks = self._select_chunks(chunks, scores, max_length, topn)

            # 计算压缩比
            compressed_text = "".join(selected_chunks)
            compression_ratio = len(compressed_text) / len(doc)

            # 记录压缩信息
            self.logger.info(f"Compression completed:")
            self.logger.info(f"- Original length: {len(doc)}")
            self.logger.info(f"- Compressed length: {len(compressed_text)}")
            self.logger.info(f"- Compression ratio: {compression_ratio:.2%}")
            self.logger.info(f"- Number of chunks: {len(selected_chunks)}")

            return CompressionResult(
                compressed_text=compressed_text,
                compression_ratio=compression_ratio,
                selected_chunks=selected_chunks,
                topic_scores=scores,
            )

        except Exception as e:
            self.logger.error(f"Compression failed: {str(e)}")
            raise
