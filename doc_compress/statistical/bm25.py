import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/" + ".." + "/" + ".."))


from typing import List, Optional
import numpy as np
from collections import Counter
import math

from util.doc_chunk import DocChunker
from util.tokenizer import tokenize_text
from util.log_util import logger
from dataclasses import dataclass
from util.time_util import calculate_execution_time

@dataclass
class CompressionResult:
    """压缩结果数据类"""

    compressed_text: str  # 压缩后的文本
    compression_ratio: float  # 压缩比例
    selected_chunks: List[str]  # 选中的文本块
    topic_scores: np.ndarray  # 主题得分


class BM25Compressor:
    """BM25-based document compression class.

    This class implements document compression using BM25 scoring, supporting both
    query-based and query-less compression modes.

    Attributes:
        k1: BM25 parameter for term frequency scaling
        b: BM25 parameter for length normalization
    """

    def __init__(self,doc_chunker,tokenize_func, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25Compressor.

        Args:
            k1: BM25 parameter for term frequency scaling (default: 1.5)
            b: BM25 parameter for length normalization (default: 0.75)
        """
        self.doc_chunker = doc_chunker
        self.tokenize_func = tokenize_func  # tokenize_func: Function that takes text and returns token list
        self.k1 = k1
        self.b = b
        self.logger = logger

    def _calculate_bm25(
        self, chunks: List[str], query: Optional[str] = None
    ) -> np.ndarray:
        """Calculate BM25 scores for document chunks.

        Args:
            chunks: List of text chunks
            query: Optional query text for guided compression

        Returns:
            numpy.ndarray: BM25 scores for each chunk
        """
        # 对所有块进行分词
        chunk_tokens = [self.tokenize_func(chunk) for chunk in chunks]

        # 计算每个块的长度和平均长度
        chunk_lengths = np.array([len(tokens) for tokens in chunk_tokens])
        avg_chunk_length = np.mean(chunk_lengths)

        # 计算词频
        chunk_token_counts = [Counter(tokens) for tokens in chunk_tokens]

        if query:
            # 使用查询词计算BM25
            query_tokens = self.tokenize_func(query)
            query_token_set = set(query_tokens)

            # 计算查询词的IDF
            idf = {}
            for token in query_token_set:
                doc_count = sum(1 for counts in chunk_token_counts if token in counts)
                idf[token] = math.log(
                    (len(chunks) - doc_count + 0.5) / (doc_count + 0.5) + 1
                )

            # 计算每个块的BM25分数
            scores = np.zeros(len(chunks))
            for i, counts in enumerate(chunk_tokens):
                chunk_score = 0
                length_norm = (
                    1 - self.b + self.b * (chunk_lengths[i] / avg_chunk_length)
                )

                for token in query_token_set:
                    tf = chunk_token_counts[i].get(token, 0)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * length_norm
                    chunk_score += idf[token] * (numerator / denominator)

                scores[i] = chunk_score
        else:
            # 不使用查询词，计算所有词的BM25
            # 获取所有唯一词
            all_tokens = set().union(*[set(tokens) for tokens in chunk_tokens])

            # 计算IDF
            idf = {}
            for token in all_tokens:
                doc_count = sum(1 for counts in chunk_token_counts if token in counts)
                idf[token] = math.log(
                    (len(chunks) - doc_count + 0.5) / (doc_count + 0.5) + 1
                )

            # 计算每个块的BM25总分
            scores = np.zeros(len(chunks))
            for i, counts in enumerate(chunk_tokens):
                chunk_score = 0
                length_norm = (
                    1 - self.b + self.b * (chunk_lengths[i] / avg_chunk_length)
                )

                for token, count in chunk_token_counts[i].items():
                    numerator = count * (self.k1 + 1)
                    denominator = count + self.k1 * length_norm
                    chunk_score += idf[token] * (numerator / denominator)

                scores[i] = chunk_score

        return scores

    def _get_chunks_by_length(
        self, chunks: List[str], scores: np.ndarray, max_length: int
    ) -> List[str]:
        """Get chunks within max_length constraint.

        Args:
            chunks: List of text chunks
            scores: BM25 scores for chunks
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

    def _get_chunks_by_topn(
        self, chunks: List[str], scores: np.ndarray, topn: int
    ) -> List[str]:
        """Get top N chunks by BM25 score.

        Args:
            chunks: List of text chunks
            scores: BM25 scores for chunks
            topn: Number of top chunks to select

        Returns:
            List[str]: Selected top N chunks
        """
        # 获取前N个最高分数的索引
        top_indices = np.argsort(-scores)[:topn]

        # 按原始顺序返回选中的块
        return [chunks[i] for i in sorted(top_indices)]

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

    @calculate_execution_time("compress")
    def compress(
        self,
        doc: str,
        max_length: Optional[int] = None,
        topn: Optional[int] = None,
        query: Optional[str] = None,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ) -> str:
        """Compress document using BM25 scoring.

        Args:
            doc: Input document text
            max_length: Maximum length of compressed document
            topn: Number of top chunks to select
            query: Optional query text for guided compression
            k1: Optional override for k1 parameter
            b: Optional override for b parameter

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

        # 临时更新BM25参数（如果提供）
        original_k1, original_b = self.k1, self.b
        if k1 is not None:
            self.k1 = k1
        if b is not None:
            self.b = b

        try:
            # 文档分块
            chunks = self.doc_chunker.batch_chunk([doc], return_counts=False)[0]
            if not chunks:
                self.logger.log_info("Document chunking produced no chunks")
                return ""

            # 计算BM25分数
            scores = self._calculate_bm25(chunks, query)

            selected_chunks = self._select_chunks(chunks, scores, max_length, topn)

            compressed_text = "".join(selected_chunks)
            compression_ratio = len(compressed_text) / len(doc)

            self.logger.log_info(f"Compression ratio: {compression_ratio:.2%}")
            self.logger.log_info(f"BM25 parameters: k1={self.k1}, b={self.b}")

            return CompressionResult(
                compressed_text=compressed_text,
                compression_ratio=compression_ratio,
                selected_chunks=selected_chunks,
                topic_scores=scores,
            )

        finally:
            # 恢复原始BM25参数
            self.k1, self.b = original_k1, original_b
