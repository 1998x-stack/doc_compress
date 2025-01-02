import argparse
import os
import time
import numpy as np


from doc_compress.statistical.tfidf import TFIDFCompressor
from doc_compress.statistical.bm25 import BM25Compressor
from util.embeddingmodel import EmbeddingModel, EmbeddingClient

parser = argparse.ArgumentParser(description="Test TF-IDF and BM25 Document Compression")
parser.add_argument('compressor', type=str, choices=['tfidf', 'bm25'], help="Compressor type")

args=parser.parse_args()


file_path = "data/test_data/zh/"
file = "红楼梦.txt"


with open(file_path + file, "r", encoding="utf-8") as f:
    doc = f.read()

query = "葬花"


if args.compressor == 'bm25':
    compressor = BM25Compressor(k1=1)
elif args.compressor == 'tfidf':
    compressor = TFIDFCompressor()

# 测试压缩文档，按最大长度选择块
compressed_doc_length = compressor.compress(doc, query=query, max_length=100)
print("Compressed Document by Length:")
print(compressed_doc_length)
print("\n")

# 测试压缩文档，选择前 2 个块
compressed_doc_topn = compressor.compress(doc, query=query, topn=2)
print("Compressed Document by TopN:")
print(compressed_doc_topn)



TEST_URL = "https://ai-platform-cloud-proxy.polymas.com/ai/common/kb-get-embedding"
embedding_client = EmbeddingClient(embedding_url=TEST_URL)
embedding_model = EmbeddingModel(embedding_client)

embedding1=embedding_model.embed_batch([query])
embedding2=embedding_model.embed_batch([compressed_doc_length])
embedding3=embedding_model.embed_batch([compressed_doc_topn])

embedding2=np.transpose(embedding2)
embedding3=np.transpose(embedding3)


similarity1 = embedding_model.similarity(embedding1, embedding2)
similarity2 = embedding_model.similarity(embedding1, embedding3)

print(f"Similarity between query and compressed doc by length: {similarity1:.4f}")
print(f"Similarity between query and compressed doc by topN: {similarity2:.4f}")

