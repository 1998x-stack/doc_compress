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


@dataclass
class CompressionResult:
    """压缩结果数据类"""

    compressed_text: str  # 压缩后的文本
    compression_ratio: float  # 压缩比例
    selected_chunks: List[str]  # 选中的文本块
    topic_scores: np.ndarray  # 主题得分


class TFIDFCompressor:
    """TF-IDF based document compression class.

    This class implements document compression using TF-IDF scoring, supporting both
    query-based and query-less compression modes.

    """

    def __init__(self) -> None:
        """Initialize TFIDFCompressor."""

        self.doc_chunker = (
            DocChunker()
        )  # doc_chunker: Instance of DocChunker for text chunking
        self.tokenize_func = tokenize_text  # tokenize_func: Function that takes text and returns token list
        # 设置日志记录器
        self.logger = logger

    def _calculate_tfidf(
        self, chunks: List[str], query: Optional[str] = None
    ) -> np.ndarray:
        """Calculate TF-IDF scores for document chunks.

        Args:
            chunks: List of text chunks
            query: Optional query text for guided compression

        Returns:
            numpy.ndarray: TF-IDF scores for each chunk
        """
        # 对所有块进行分词
        chunk_tokens = [self.tokenize_func(chunk) for chunk in chunks]

        # 计算词频（TF）
        chunk_token_counts = [Counter(tokens) for tokens in chunk_tokens]

        if query:
            # 使用查询词计算TF-IDF
            query_tokens = self.tokenize_func(query)
            query_token_set = set(query_tokens)

            # 计算查询词的IDF
            idf = {}
            for token in query_token_set:
                doc_count = sum(1 for counts in chunk_token_counts if token in counts)
                idf[token] = math.log(len(chunks) / (1 + doc_count))

            # 计算每个块的TF-IDF分数
            scores = np.zeros(len(chunks))
            for i, counts in enumerate(chunk_token_counts):
                chunk_score = 0
                for token in query_token_set:
                    if len(chunk_tokens[i]) == 0:
                        continue
                    tf = counts.get(token, 0) / len(chunk_tokens[i])
                    chunk_score += tf * idf[token]
                scores[i] = chunk_score
        else:
            # 不使用查询词，计算所有词的TF-IDF
            # 获取所有唯一词
            all_tokens = set().union(*[set(tokens) for tokens in chunk_tokens])

            # 计算IDF
            idf = {}
            for token in all_tokens:
                doc_count = sum(1 for counts in chunk_token_counts if token in counts)
                idf[token] = math.log(len(chunks) / (1 + doc_count))

            # 计算每个块的TF-IDF总分
            scores = np.zeros(len(chunks))
            for i, counts in enumerate(chunk_token_counts):
                chunk_score = 0
                for token, count in counts.items():
                    tf = count / len(chunk_tokens[i])
                    chunk_score += tf * idf[token]
                scores[i] = chunk_score

        return scores

    def _get_chunks_by_length(
        self, chunks: List[str], scores: np.ndarray, max_length: int
    ) -> List[str]:
        """Get chunks within max_length constraint.

        Args:
            chunks: List of text chunks
            scores: TF-IDF scores for chunks
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
        """Get top N chunks by TF-IDF score.

        Args:
            chunks: List of text chunks
            scores: TF-IDF scores for chunks
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

    def compress(
        self,
        doc: str,
        query: Optional[str] = None,
        max_length: Optional[int] = None,
        topn: Optional[int] = None,
    ) -> str:
        """Compress document using TF-IDF scoring.

        Args:
            doc: Input document text
            query: Optional query text for guided compression
            max_length: Maximum length of compressed document
            topn: Number of top chunks to select

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

        # 文档分块
        chunks = self.doc_chunker.batch_chunk([doc], return_counts=False)[0]
        if not chunks:
            self.logger.log_info("Document chunking produced no chunks")
            return ""

        # 计算TF-IDF分数
        scores = self._calculate_tfidf(chunks, query)

        selected_chunks = self._select_chunks(chunks, scores, max_length, topn)

        compressed_text = "".join(selected_chunks)
        compression_ratio = len(compressed_text) / len(doc)
        self.logger.log_info(f"Compression ratio: {compression_ratio:.2%}")

        # 返回压缩后的文档
        return CompressionResult(
            compressed_text=compressed_text,
            compression_ratio=compression_ratio,
            selected_chunks=selected_chunks,
            topic_scores=scores,
        )
