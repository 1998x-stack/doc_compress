import argparse
import os
import time
import sys
import numpy as np
import pandas as pd
import chardet


from doc_compress.statistical.tfidf import TFIDFCompressor
from doc_compress.statistical.bm25 import BM25Compressor
from doc_compress.graph.textrank import TextRankCompressor
from doc_compress.graph.singlerank import SingleRankCompressor
from doc_compress.graph.positionrank import PositionRankCompressor
from doc_compress.graph.topicrank import TopicRankCompressor
from util.embeddingmodel import EmbeddingModel, EmbeddingClient
from util.doc_chunk import DocChunker
from util.tokenizer import tokenize_text


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
    if args.language == "en":
        csv_file = "en" + csv_file
        file_path = file_path + "en/"
    elif args.language == "zh":
        csv_file = "zh" + csv_file
        file_path = file_path + "zh/"

    df = pd.read_csv(file_path + csv_file)
    df = df.head(5)

    if args.compressor == "bm25":
        compressor = BM25Compressor()
    elif args.compressor == "tfidf":
        compressor = TFIDFCompressor()
    elif args.compressor == "textrank":
        compressor = TextRankCompressor(DocChunker(), tokenize_text)
    elif args.compressor == "singlerank":
        compressor = SingleRankCompressor(DocChunker(), tokenize_text)
    elif args.compressor == "positionrank":
        compressor = PositionRankCompressor(DocChunker(), tokenize_text)
    elif args.compressor == "topicrank":
        compressor = TopicRankCompressor(DocChunker(), tokenize_text)

    results = []

    for _, row in df.iterrows():
        file_name = row["File Name"]
        querys = row["Query"]  # TODO：query增长

        encoding = detect_encoding(file_path + file_name)

        with open(file_path + file_name, "r", encoding=encoding, errors="ignore") as f:
            doc = f.read()

        query_terms = querys.split()
        # query_terms = [""]
        for query in query_terms:  # TODO: 将返回值进行封装
            compressresult1 = compressor.compress(doc, query=query, max_length=300)
            print(
                f"Compressed Document by Length for {file_name} with query '{query}':"
            )
            print(compressresult1.compressed_text)

            compressresult2 = compressor.compress(doc, query=query, topn=2)
            print(f"Compressed Document by TopN for {file_name} with query '{query}':")
            print(compressresult2.compressed_text)

            similarity1, similarity2 = evaluate_similarity(
                query, compressresult1.compressed_text, compressresult2.compressed_text
            )
            results.append(
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
            )

    results_df = pd.DataFrame(results)
    output_file_name = f"compressed_results_{args.compressor}_{args.language}.csv"
    results_df.to_csv(output_file_name, index=False)


csv_file = "_query.csv"
file_path = "data/test_data/"


sys.argv = ["test.py", "bm25", "zh"]
parser = argparse.ArgumentParser(
    description="Test TF-IDF and BM25 Document Compression"
)
parser.add_argument(
    "compressor",
    type=str,
    choices=["tfidf", "bm25", "textrank", "singlerank", "positionrank", "topicrank"],
    help="Compressor type",
)

parser.add_argument(
    "language", type=str, choices=["en", "zh"], help="Language of the documents"
)
args = parser.parse_args()


# 执行批量处理
batch_process_csv(csv_file, file_path)
