from __future__ import annotations

import hashlib
import math
import uuid
from typing import Iterable

import chromadb

from utils.config import settings


class HashEmbeddings:
    """Small local embedding function so the demo works without a second API key.

    ChromaDB receives deterministic normalized vectors. For production, replace this
    with OpenAI, Voyage, Cohere, or a sentence-transformers embedding function.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class ChromaResearchStore:
    def __init__(self, persist_directory: str | None = None, collection_name: str | None = None) -> None:
        self.client = chromadb.PersistentClient(path=persist_directory or settings.chroma_dir)
        self.embeddings = HashEmbeddings()
        self.collection = self.client.get_or_create_collection(
            name=collection_name or settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: Iterable[dict]) -> list[str]:
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []

        for doc in documents:
            text = (doc.get("content") or "").strip()
            if not text:
                continue
            ids.append(str(uuid.uuid4()))
            texts.append(text)
            metadatas.append(
                {
                    "title": doc.get("title", "Untitled"),
                    "source": doc.get("source", "unknown"),
                    "url": doc.get("url", ""),
                    "kind": doc.get("kind", "web"),
                }
            )

        if texts:
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=self.embeddings.embed_documents(texts))
        return ids

    def similarity_search(self, query: str, k: int = 8) -> list[dict]:
        results = self.collection.query(query_embeddings=[self.embeddings.embed(query)], n_results=k)
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            {"content": doc, "metadata": metadata or {}, "distance": distance}
            for doc, metadata, distance in zip(docs, metadatas, distances, strict=False)
        ]
