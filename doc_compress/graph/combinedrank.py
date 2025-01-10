import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/" + ".." + "/" + ".."))


from typing import List, Optional, Dict, Tuple
import numpy as np
import networkx as nx
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from dataclasses import dataclass
from doc_compress.graph.textrank import TextRankCompressor
from util.time_util import calculate_execution_time


@dataclass
class CompressionResult:
    """压缩结果数据类"""

    compressed_text: str  # 压缩后的文本
    compression_ratio: float  # 压缩比例
    selected_chunks: List[str]  # 选中的文本块
    topic_scores: List[float]  # 主题得分


class CombinedRankCompressor(TextRankCompressor):
    """综合基于SingleRank和PositionRank算法的文档压缩类。

    该类通过继承TextRankCompressor，并根据用户需求引入TF-IDF权重和位置权重，
    从而增强文档压缩效果。支持带查询和不带查询两种压缩模式，可以基于
    最大长度或选择前N个块进行压缩。

    Attributes:
        doc_chunker: 文档分块器实例
        tokenize_func: 文本分词函数
        window_size: 共现窗口大小
        damping_factor: PageRank阻尼系数
        min_edge_weight: 最小边权重阈值
        tfidf_vectorizer: TF-IDF向量化器
        use_tfidf: 是否启用TF-IDF权重
        use_position: 是否启用位置权重
        position_weight_decay: 位置权重的衰减因子
    """

    def __init__(
        self,
        doc_chunker,
        tokenize_func,
        window_size: int = 3,
        damping_factor: float = 0.85,
        min_edge_weight: float = 0.1,
        use_tfidf: bool = True,
        use_position: bool = True,
        position_weight_decay: float = 0.95,  # 可定制的衰减因子
        tfidf_kwargs: Optional[Dict] = None,  # 传递给TfidfVectorizer的其他参数
    ) -> None:
        """初始化CombinedRankCompressor。

        Args:
            doc_chunker: 文档分块器实例
            tokenize_func: 分词函数，输入文本返回token列表
            window_size: 共现窗口大小，默认为3
            damping_factor: PageRank阻尼系数，默认为0.85
            min_edge_weight: 最小边权重阈值，默认为0.1
            use_tfidf: 是否启用TF-IDF权重，默认为True
            use_position: 是否启用位置权重，默认为True
            position_weight_decay: 位置权重的衰减因子，默认为0.95
            tfidf_kwargs: 传递给TfidfVectorizer的其他参数
        """
        super().__init__(
            doc_chunker=doc_chunker,
            tokenize_func=tokenize_func,
            window_size=window_size,
            damping_factor=damping_factor,
            min_edge_weight=min_edge_weight,
        )
        self.use_tfidf = use_tfidf
        self.use_position = use_position

        if self.use_tfidf:
            if tfidf_kwargs is None:
                tfidf_kwargs = {}
            self.tfidf_vectorizer = TfidfVectorizer(
                tokenizer=tokenize_func,
                lowercase=False,  # 保持原始大小写
                **tfidf_kwargs,
            )
        else:
            self.tfidf_vectorizer = None

        if self.use_position:
            self.position_weight_decay = position_weight_decay
        else:
            self.position_weight_decay = None

        self.logger = logging.getLogger(__name__)

    def _calculate_tfidf_weights(self, chunks: List[str]) -> Dict[str, float]:
        """计算文档块中词语的TF-IDF权重。

        Args:
            chunks: 文档块列表

        Returns:
            Dict[str, float]: 词语到TF-IDF权重的映射
        """
        if not self.use_tfidf or not self.tfidf_vectorizer:
            return {}

        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunks)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            avg_weights = np.mean(tfidf_matrix.toarray(), axis=0)
            return dict(zip(feature_names, avg_weights))
        except Exception as e:
            self.logger.error(f"TF-IDF计算失败: {str(e)}")
            return {}

    def _calculate_position_weights(self, tokens: List[str]) -> Dict[str, float]:
        """计算词语的位置权重。

        Args:
            tokens: 分词后的token列表

        Returns:
            Dict[str, float]: 词语到位置权重的映射
        """
        if not self.use_position or self.position_weight_decay is None:
            return {}

        position_weights = defaultdict(float)
        token_positions = defaultdict(list)

        # 记录每个词的所有出现位置
        for pos, token in enumerate(tokens, 1):
            token_positions[token].append(pos)

        # 计算每个词的位置权重
        for token, positions in token_positions.items():
            # 位置权重 = Σ(1/position) * decay^(position_index)
            weight = sum(
                (1.0 / pos) * (self.position_weight_decay ** (pos - 1))
                for pos in positions
            )
            position_weights[token] = weight

        return dict(position_weights)

    def _calculate_chunk_weights(
        self, chunks: List[str]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]],List[List[str]]]:
        """计算全局TF-IDF权重和每个块的词语位置权重。

        Args:
            chunks: 文档块列表

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]],List[List[str]]]:
                全局TF-IDF权重映射，每个块中词语的位置权重列表，每个块的token列表
        """
        tfidf_weights = self._calculate_tfidf_weights(chunks)
        chunk_position_weights = []
        chunk_tokens = []
        if self.use_position:
            for chunk in chunks:
                tokens = self.tokenize_func(chunk)
                chunk_tokens.append(tokens)
                weights = self._calculate_position_weights(tokens)
                chunk_position_weights.append(weights)
        else:
            # 如果不使用位置权重，则每个块的权重为空字典
            chunk_position_weights = [{} for _ in chunks]
            chunk_tokens = [self.tokenize_func(chunk) for chunk in chunks]

        return tfidf_weights, chunk_position_weights, chunk_tokens

    def _build_weighted_graph(
        self, chunks: List[str], query: Optional[str] = None
    ) -> Tuple[nx.Graph, Dict[str, float], List[Dict[str, float]]]:
        """构建加权图。

        使用TF-IDF权重和位置权重结合共现关系构建图，
        如果提供查询，则增加与查询相关词语的权重。

        Args:
            chunks: 文档块列表
            query: 可选的查询文本

        Returns:
            Tuple[nx.Graph, Dict[str, float], List[Dict[str, float]]]:
                加权图，全局TF-IDF权重映射，每个块的词语位置权重列表
        """
        graph = nx.Graph()

        # 计算全局TF-IDF权重和每个块的词语位置权重
        tfidf_weights, chunk_position_weights, chunk_tokens = (
            self._calculate_chunk_weights(chunks)
        )

        # 处理查询词（如果有）
        query_tokens = set()
        if query:
            query_tokens = set(self.tokenize_func(query))

        # 统计共现关系并构建图
        for chunk_idx, tokens in enumerate(chunk_tokens):
            tfidf_weight = tfidf_weights if self.use_tfidf else {}
            position_weights = (
                chunk_position_weights[chunk_idx] if self.use_position else {}
            )

            for i in range(len(tokens)):
                word1 = tokens[i]
                for j in range(i + 1, min(i + self.window_size + 1, len(tokens))):
                    word2 = tokens[j]
                    if word1 == word2:
                        continue

                    # 计算边权重
                    weight = 1.0
                    if self.use_tfidf:
                        weight *= tfidf_weight.get(word1, 0) * tfidf_weight.get(
                            word2, 0
                        )
                    if self.use_position:
                        weight *= 1.0 / (j - i)  # 距离衰减

                    # 如果是查询相关词，增加权重
                    if query and (word1 in query_tokens or word2 in query_tokens):
                        weight *= 2.0

                    if weight >= self.min_edge_weight:
                        if graph.has_edge(word1, word2):
                            graph[word1][word2]["weight"] += weight
                        else:
                            graph.add_edge(word1, word2, weight=weight)

        return graph, tfidf_weights, chunk_position_weights, chunk_tokens

    def _calculate_chunk_scores(
        self, chunks: List[str], query: Optional[str] = None
    ) -> List[float]:
        """计算文档块的综合得分。

        Args:
            chunks: 文档块列表
            query: 可选的查询文本

        Returns:
            List[float]: 每个块的得分
        """
        # 构建加权图
        graph, tfidf_weights, chunk_position_weights, chunk_tokens = (
            self._build_weighted_graph(chunks, query)
        )

        # 如果图为空，返回零分数组
        if not graph:
            self.logger.warning("Empty graph constructed")
            return [0.0 for _ in chunks]

        # 计算PageRank得分
        try:
            pagerank_scores = nx.pagerank(
                graph, alpha=self.damping_factor, weight="weight"
            )
        except Exception as e:
            self.logger.error(f"PageRank计算失败: {str(e)}")
            return [0.0 for _ in chunks]

        # 计算每个块的得分
        scores = []

        for i, tokens in enumerate(chunk_tokens):
            unique_tokens = set(tokens)
            chunk_score = 0.0

            # 累加块中每个词的PageRank得分
            for token in unique_tokens:
                token_score = pagerank_scores.get(token, 0)
                if self.use_tfidf:
                    token_score *= tfidf_weights.get(token, 1)
                if self.use_position:
                    token_score *= chunk_position_weights[i].get(token, 1)
                chunk_score += token_score

            if query:
                # 如果有查询，增加与查询相关词的得分
                query_tokens = set(self.tokenize_func(query))
                relevance = sum(
                    2 * pagerank_scores.get(token, 0)
                    for token in unique_tokens & query_tokens
                )
                chunk_score += relevance

            # 归一化得分
            if unique_tokens:
                chunk_score /= len(unique_tokens)

            scores.append(chunk_score)

        return scores

    def _select_chunks(
        self,
        chunks: List[str],
        scores: List[float],
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

        selected_indices = set()

        if max_length is not None:
            # 基于最大长度选择
            current_length = 0
            for idx in sorted_indices:
                if current_length + lengths[idx] <= max_length:
                    selected_indices.add(idx)
                    current_length += lengths[idx]
                else:
                    continue  # 尝试下一个块

        if topn is not None:
            # 选择前N个块
            topn_indices = sorted_indices[:topn]
            selected_indices.update(topn_indices)

        # 按原始顺序返回选中的块
        selected_indices = sorted(selected_indices)
        return [chunks[i] for i in selected_indices]

    @calculate_execution_time("compress")
    def compress(
        self,
        doc: str,
        query: Optional[str] = None,
        max_length: Optional[int] = None,
        topn: Optional[int] = None,
    ) -> CompressionResult:
        """使用CombinedRank算法压缩文档。

        Args:
            doc: 输入文档文本
            query: 可选的查询文本
            max_length: 压缩后文档的最大长度
            topn: 选择得分最高的前N个块

        Returns:
            CompressionResult: 压缩结果

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
            chunks, _ = self.doc_chunker.batch_chunk(text_list=[doc])
            chunks = chunks[0]
            if not chunks:
                self.logger.warning("文档分块结果为空")
                return CompressionResult(
                    compressed_text="",
                    compression_ratio=0.0,
                    selected_chunks=[],
                    topic_scores=[],
                )

            # 计算块得分
            scores = self._calculate_chunk_scores(chunks, query)

            # 选择块
            selected_chunks = self._select_chunks(chunks, scores, max_length, topn)

            # 生成压缩文本
            compressed_text = "".join(selected_chunks)

            # 计算压缩比率
            compression_ratio = len(compressed_text) / len(doc) if len(doc) > 0 else 0.0

            # 日志记录
            self.logger.info(f"压缩比率: {compression_ratio:.2%}")
            self.logger.info(
                f"CombinedRank参数: window_size={self.window_size}, "
                f"damping_factor={self.damping_factor}, "
                f"use_tfidf={self.use_tfidf}, use_position={self.use_position}"
            )

            # 返回压缩结果
            return CompressionResult(
                compressed_text=compressed_text,
                compression_ratio=compression_ratio,
                selected_chunks=selected_chunks,
                topic_scores=scores,
            )

        except Exception as e:
            self.logger.error(f"文档压缩过程中发生错误: {str(e)}")
            raise


if __name__ == "__main__":
    import logging
    from util.doc_chunk import DocChunker
    from util.tokenizer import tokenize_text

    # 设置日志记录级别
    logging.basicConfig(level=logging.INFO)

    # 实例化一个简单的文档分块器
    doc_chunker = DocChunker()

    # 实例化CombinedRankCompressor
    compressor = CombinedRankCompressor(
        doc_chunker=doc_chunker,
        tokenize_func=tokenize_text,
        window_size=30,
        damping_factor=0.85,
        min_edge_weight=0.1,
        use_tfidf=True,
        use_position=True,
        position_weight_decay=0.95,
        tfidf_kwargs={"stop_words": "english"},  # 示例：移除英文停用词
    )

    # 示例文档
    document = (
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence. "
        "NLP is concerned with the interactions between computers and human language. "
        "In particular, how to program computers to process and analyze large amounts of natural language data."
    )

    # 查询词
    query = "artificial intelligence"

    # 压缩文档
    result = compressor.compress(doc=document, query=query, max_length=200, topn=2)

    # 输出结果
    print("压缩后的文本:")
    print(result.compressed_text)
    print(f"压缩比率: {result.compression_ratio:.2%}")
    print("选中的文本块:")
    for chunk in result.selected_chunks:
        print(f"- {chunk}")
    print("主题得分:")
    print(result.topic_scores)
