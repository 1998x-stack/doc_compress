import argparse
import os
import time
import sys
import numpy as np
import pandas as pd
import chardet
import json
import jieba



from doc_compress.statistical.tfidf import TFIDFCompressor
from doc_compress.statistical.bm25 import BM25Compressor
from doc_compress.graph.combinedrank import CombinedRankCompressor
from doc_compress.graph.textrank import TextRankCompressor
from doc_compress.graph.singlerank import SingleRankCompressor
from doc_compress.graph.positionrank import PositionRankCompressor
from doc_compress.graph.topicrank import TopicRankCompressor
from util.embeddingmodel import EmbeddingModel, EmbeddingClient
from util.doc_chunk import DocChunker
from util.tokenizer import tokenize_text
from config.config import Config


def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
        return result["encoding"]


def evaluate_similarity(
    query,
    compressed_doc_length,
    compressed_doc_topn,
    embedding_url="https://ai-platform-cloud-proxy.polymas.com/ai/common/kb-get-embedding",
):
    embedding_client = EmbeddingClient(embedding_url=embedding_url)
    embedding_model = EmbeddingModel(embedding_client)

    embedding1 = embedding_model.embed_batch([query])
    embedding2 = embedding_model.embed_batch([compressed_doc_length])
    embedding3 = embedding_model.embed_batch([compressed_doc_topn])

    embedding2 = np.transpose(embedding2)
    embedding3 = np.transpose(embedding3)

    similarity1 = embedding_model.similarity(embedding1, embedding2)
    similarity2 = embedding_model.similarity(embedding1, embedding3)

    # 输出相似度
    print(f"Similarity between query and compressed doc by length: {similarity1:.4f}")
    print(f"Similarity between query and compressed doc by topN: {similarity2:.4f}")

    return similarity1, similarity2


def batch_process_csv(csv_file, file_path):
    config = Config()
    if args.language == "en":
        csv_file = "en" + csv_file
        file_path = file_path + "en/"
        sentence_length = config.get_config("token_length_en")

    elif args.language == "zh":
        csv_file = "zh" + csv_file
        file_path = file_path + "zh/"
        sentence_length = config.get_config("token_length_zh")

    topn = config.get_config("topN")
    max_length = config.get_config("compressed_length")
    df = pd.read_csv(file_path + csv_file)
    use_tfidf = config.get_config("graph")["use_tfidf"]
    use_position = config.get_config("graph")["use_position"]
    window_size = config.get_config("graph")["window_size"]
    
    # df = df.head(20)

    if args.compressor == "bm25":
        compressor = BM25Compressor(
            DocChunker(max_sentence_length=sentence_length), tokenize_text
        )
    elif args.compressor == "tfidf":
        compressor = TFIDFCompressor(
            DocChunker(max_sentence_length=sentence_length), tokenize_text
        )
    elif args.compressor == "textrank":
        compressor = CombinedRankCompressor(
            DocChunker(max_sentence_length=sentence_length),
            tokenize_text,
            window_size=window_size,
            use_tfidf=False,
            use_position=False,
        )
    elif args.compressor == "singlerank":
        compressor = CombinedRankCompressor(
            DocChunker(max_sentence_length=sentence_length),
            tokenize_text,
            window_size=window_size,
            use_tfidf=True,
            use_position=False,
        )
    elif args.compressor == "positionrank":
        compressor = CombinedRankCompressor(
            DocChunker(max_sentence_length=sentence_length),
            tokenize_text,
            window_size=window_size,
            use_tfidf=False,
            use_position=True,
        )
    elif args.compressor == "pos+tfidfrank":
        compressor = CombinedRankCompressor(
            DocChunker(max_sentence_length=sentence_length),
            tokenize_text,
            window_size=window_size,
            use_tfidf=True,
            use_position=True,
        )
    else:
        compressor = CombinedRankCompressor(
            DocChunker(max_sentence_length=sentence_length),
            tokenize_text,
            window_size=window_size,
            use_tfidf=use_tfidf,
            use_position=use_position,
        )

    results = []

    for _, row in df.iterrows():
        file_name = row["File Name"]
        querys = row["Query"]
        base_name, end = os.path.splitext(file_name)
        for i in range(1, 11):
            file_name = base_name + "_part" + str(i) + end
            encoding = detect_encoding(file_path + file_name)
            with open(
                file_path + file_name, "r", encoding=encoding, errors="ignore"
            ) as f:
                doc = f.read()

            query_terms = [
                query.strip() for query in querys.split("?") if query.strip()
            ]
            # query_terms = [""]
            for query in query_terms:
                compressresult1 = compressor.compress(
                    doc, query=query, max_length=max_length
                )
                print(
                    f"Compressed Document by Length for {file_name} with query '{query}':"
                )
                print(compressresult1.compressed_text)

                compressresult2 = compressor.compress(doc, query=query, topn=topn)
                print(
                    f"Compressed Document by TopN for {file_name} with query '{query}':"
                )
                print(compressresult2.compressed_text)

                similarity1, similarity2 = evaluate_similarity(
                    query,
                    compressresult1.compressed_text,
                    compressresult2.compressed_text,
                )

                result = [
                    {
                        "File Name": file_name,
                        "Query Term": query,
                        "Compressed by Length": compressresult1.compressed_text,
                        "Compressed by TopN": compressresult2.compressed_text,
                        "Similarity by Length": similarity1,
                        "Similarity by TopN": similarity2,
                        "Compression Ratio by Length": compressresult1.compression_ratio,
                        "Compression Ratio by TopN": compressresult2.compression_ratio,
                    }
                ]

                results.append(result)

    output_dir = "data/test_result"
    output_file_name = (
        f"{output_dir}/compressed_results_{args.compressor}_{args.language}.json"
    )
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    jieba.initialize()
    csv_file = "_file.csv"
    file_path = "data/test_data/split_files/"

    sys.argv = ["test.py", "textrank", "en"]

    parser = argparse.ArgumentParser(description="Test Document Compression")
    parser.add_argument(
        "compressor",
        type=str,
        choices=[
            "tfidf",
            "bm25",
            "textrank",
            "singlerank",
            "positionrank",
            "topicrank",
            "pos_tfidfrank"
        ],
        help="Compressor type",
    )

    parser.add_argument(
        "language", type=str, choices=["en", "zh"], help="Language of the documents"
    )
    args = parser.parse_args()

    # 执行批量处理
    batch_process_csv(csv_file, file_path)
