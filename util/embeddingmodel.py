import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/" + ".."))

import numpy as np
from typing import List, Optional
import requests


class EmbeddingModel:
    """A placeholder embedding model class.

    This class simulates an embedding model with:
    - A `embed_batch` method to embed a batch of texts.
    - A `similarity` method to compute similarity between two embeddings.

    Note:
        In a real-world scenario, this would load a specific model
        (e.g., a SentenceTransformer model or a proprietary large language model)
        based on the model_name. Here, we just simulate embeddings with random vectors.
    """

    def __init__(self, embedding_client, embedding_dim: int = 128):
        # 中文注释: 初始化嵌入模型，实际实现中可加载相应的模型文件。
        self.embedding_client = embedding_client
        self.embedding_dim = embedding_dim

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts and return a list of embedding vectors.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[np.ndarray]: List of embedding vectors.
        """
        # 中文注释: 返回模拟的随机嵌入向量，实际情况需调用真实模型方法。
        # return [np.random.rand(self.embedding_dim).astype(np.float32) for _ in texts]
        # embeddings = embedding_api(texts)
        # 每十个文本一组，避免一次请求过多文本
        batch_size = 10
        embeddings = []
        if(len(texts) < batch_size):
            response = self.embedding_client.get_embeddings(texts)
            embeddings = response.get("data", {}).get("resultList", [])
            return [np.array(embedding).astype(np.float32) for embedding in embeddings]
        else:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                response = self.embedding_client.get_embeddings(batch_texts)
                batch_embeddings = response.get("data", {}).get("resultList", [])
                embeddings.extend(batch_embeddings)
            return [np.array(embedding).astype(np.float32) for embedding in embeddings]

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1 (np.ndarray): Embedding vector 1.
            embedding2 (np.ndarray): Embedding vector 2.

        Returns:
            float: Cosine similarity between the two embeddings.
        """
        return float(np.dot(embedding1, embedding2))


class EmbeddingClient:
    """
    A client for interacting with the embedding service API.

    This class handles sending text data to the embedding API and retrieving the generated embeddings.
    It uses the URL from the environment variable `EMBEDDING_URL` for the API endpoint.

    Attributes:
        url (str): The base URL of the embedding service API.
        headers (dict): The headers required by the API for the request.
    """

    def __init__(self, embedding_url: str = None):
        """
        Initializes the EmbeddingClient with the URL from the environment variable or provided as an argument.

        Args:
            embedding_url (str): The embedding API URL. Defaults to None, in which case the URL is retrieved from the environment variable 'EMBEDDING_URL'.
        """
        self.url = embedding_url or os.getenv("EMBEDDING_URL")
        if not self.url:
            raise ValueError(
                "Embedding URL must be provided either as an argument or via environment variable 'EMBEDDING_URL'."
            )

        # Set headers for the HTTP request
        self.headers = {
            "Content-Type": "application/json",
            "Cookie": "acw_tc=0a5cc91217346004928187547ede31ab8900fa93156591e93b26d29b8a9da9",  # Cookie should be secure and configurable
        }

    def get_embeddings(
        self,
        text_list: List[str],
        model: str = "m3e",
        version: str = "m3e",
        unique_id: str = "test",
    ) -> dict:
        """
        Sends the text list to the embedding API and retrieves the embeddings.

        Args:
            text_list (List[str]): A list of text strings to be embedded.
            model (str): The model to use for embedding generation (default: "m3e").
            version (str): The version of the model to use (default: "m3e").
            unique_id (str): A unique identifier for the request (default: "test").

        Returns:
            dict: The response from the API containing the generated embeddings or an error message.

        Raises:
            ValueError: If the API response indicates an error.
        """
        payload = {
            "model": model,
            "textList": text_list,
            "version": version,
            "uniqueId": unique_id,
        }

        # 发送请求并获取响应
        try:
            response = requests.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()  # Check if the request was successful
            response_data = response.json()

            # 检查API返回的状态是否正确
            if "error" in response_data:
                raise ValueError(f"API Error: {response_data['error']}")

            return response_data
        except requests.RequestException as e:
            # 捕捉请求中的任何异常，打印日志信息
            raise ValueError(f"Request failed: {e}")

    def print_embeddings(self, text_list: List[str]) -> None:
        """
        Retrieves and prints the embeddings for a given list of texts.

        Args:
            text_list (List[str]): The list of text to get embeddings for.
        """
        try:
            embeddings = self.get_embeddings(text_list)
            result_list = embeddings.get("data", {}).get("resultList", [])
            print(f"Embeddings for the provided text list:")
            for text, embedding in zip(text_list, result_list):
                print(f"Text: {text}")
                print(f"Embedding: {embedding}")
        except ValueError as e:
            print(f"Error: {e}")