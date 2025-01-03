import argparse
import os
import time
import numpy as np
import pandas as pd


from doc_compress.statistical.tfidf import TFIDFCompressor
from doc_compress.statistical.bm25 import BM25Compressor
from doc_compress.graph.textrank import TextRankCompressor
from util.embeddingmodel import EmbeddingModel, EmbeddingClient


parser = argparse.ArgumentParser(
    description="Test TF-IDF and BM25 Document Compression"
)
parser.add_argument(
    "compressor", type=str, choices=["tfidf", "bm25","textrank"], help="Compressor type"
)
args = parser.parse_args()


def evaluate_similarity(query, compressed_doc_length, compressed_doc_topn, embedding_url="https://ai-platform-cloud-proxy.polymas.com/ai/common/kb-get-embedding"):
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
    df = pd.read_csv(file_path+csv_file)
    
    
    df = df.head(51)
    
    if args.compressor == "bm25":
        compressor = BM25Compressor(k1=1)
    elif args.compressor == "tfidf":
        compressor = TFIDFCompressor()
    elif args.compressor == "textrank":
        compressor = TextRankCompressor()

    results = []

    for _, row in df.iterrows():
        file_name = row['File Name']
        query = row['Query']
        
        with open(file_path + file_name, "r", encoding="utf-8") as f:
            doc = f.read()
        
        query_terms = query.split()
        for term in query_terms:
            compressed_doc_length,ratio1= compressor.compress(doc, query=term, max_length=300)
            print(f"Compressed Document by Length for {file_name} with query '{term}':")
            print(compressed_doc_length)
            
            compressed_doc_topn,ratio2= compressor.compress(doc, query=term, topn=2)
            print(f"Compressed Document by TopN for {file_name} with query '{term}':")
            print(compressed_doc_topn)
            
            similarity1,similarity2=evaluate_similarity(query, compressed_doc_length, compressed_doc_topn)
            results.append({
                'File Name': file_name,
                'Query Term': term,
                'Compressed by Length': compressed_doc_length,
                'Compressed by TopN': compressed_doc_topn,
                'Similarity by Length': similarity1,
                'Similarity by TopN': similarity2,
                'Compression Ratio by Length': ratio1,
                'Compression Ratio by TopN': ratio2
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('compressed_results.csv', index=False)

csv_file = 'en_query.csv'
file_path = 'data/test_data/en/'

# 执行批量处理
batch_process_csv(csv_file, file_path)