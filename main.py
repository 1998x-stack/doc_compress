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


def Compressor():
    config = Config()
    compressor_type = config.get_config("compressor_type")
    language = config.get_config("language")
    if language == "en":
        sentence_length = config.get_config("token_length_en")
    elif language == "zh":
        sentence_length = config.get_config("token_length_zh")
    use_tfidf = config.get_config("graph")["use_tfidf"]
    use_position = config.get_config("graph")["use_position"]
    window_size = config.get_config("graph")["window_size"]
    if compressor_type == "bm25":
        compressor = BM25Compressor(
            DocChunker(max_sentence_length=sentence_length), tokenize_text
        )
    elif compressor_type == "tfidf":
        compressor = TFIDFCompressor(
            DocChunker(max_sentence_length=sentence_length), tokenize_text
        )
    elif compressor_type == "textrank":
        compressor = CombinedRankCompressor(
            DocChunker(max_sentence_length=sentence_length),
            tokenize_text,
            window_size=window_size,
            use_tfidf=False,
            use_position=False,
        )
    elif compressor_type == "singlerank":
        compressor = CombinedRankCompressor(
            DocChunker(max_sentence_length=sentence_length),
            tokenize_text,
            window_size=window_size,
            use_tfidf=True,
            use_position=False,
        )
    elif compressor_type == "positionrank":
        compressor = CombinedRankCompressor(
            DocChunker(max_sentence_length=sentence_length),
            tokenize_text,
            window_size=window_size,
            use_tfidf=False,
            use_position=True,
        )
    elif compressor_type == "pos+tfidfrank":
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

    return compressor


def Compress(csv_file, file_path):
    compressor = Compressor()
    config = Config()
    compressor_type = config.get_config("compressor_type")
    language = config.get_config("language")
    file_path=file_path+language+'/'
    topn = config.get_config("topN")
    max_length = config.get_config("compressed_length")
    use_query = config.get_config("use_query")
    df = pd.read_csv(csv_file)
    results = []

    for _, row in df.iterrows():
        file_name = row["File Name"]
        if use_query:
            querys = row["Query"]
            query_terms = [
                query.strip() for query in querys.split("?") if query.strip()
            ]
        else:
            query_terms = [""]
        base_name, end = os.path.splitext(file_name)
        for i in range(1, 11):
            file_name = base_name + "_part" + str(i) + end
            encoding = detect_encoding(file_path + file_name)
            with open(
                file_path + file_name, "r", encoding=encoding, errors="ignore"
            ) as f:
                doc = f.read()

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
        f"{output_dir}/compressed_results_{compressor_type}_{language}.json"
    )
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    csv_file = "data/test_data/split_files/en/en_file.csv"
    file_path = "data/test_data/split_files/"
    Compress(csv_file,file_path)
