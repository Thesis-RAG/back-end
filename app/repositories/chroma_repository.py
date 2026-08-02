"""
ChromaDB vector store repository.
Collections use hnsw:space=cosine; distances are in [0, 1] (1 − cosine_sim).
Keyword search uses Chroma's where_document=$contains filter to avoid full-corpus scans.
"""
from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

import chromadb  # noqa: E402

from app.core.config import settings

try:
    from pyvi import ViTokenizer
except ImportError:
    ViTokenizer = None

# Vietnamese stopwords filtered out before keyword tokenisation. Includes
# both generic function words and interrogative-phrase words ("là gì",
# "thế nào", "ở đâu", ...) — a question word carries no BM25 matching
# value of its own (every document and every question contains some of
# them), so leaving them in only dilutes term-frequency signal for the
# words that actually distinguish one query from another.
_VN_STOPWORDS = {
    "là", "của", "và", "có", "được", "này", "đó", "các", "cho", "với",
    "tại", "về", "như", "từ", "trong", "khi", "để", "theo", "những",
    "một", "đã", "sẽ", "thì", "mà", "hay", "hoặc", "nên", "vì", "do",
    "bị", "bởi", "ra", "vào", "lên", "xuống", "đang", "rất", "cũng",
    "không", "còn", "nữa", "nào", "gì", "ai", "sao", "bao", "nhiêu",
    "thế", "ở", "đâu", "vậy", "à", "ư", "nhé", "nhỉ",
}

# Stripped from both ends of every token after segmentation — punctuation
# left attached (eg a trailing "?" glued onto the last word of a question:
# "An?") makes that token match nothing in the corpus even when the bare
# word appears there, silently zeroing out that word's entire contribution
# to BM25 scoring. \W with Unicode strings already treats Vietnamese
# diacritic letters as word characters, so this only strips true punctuation.
_PUNCT_STRIP_RE = re.compile(r"^\W+|\W+$", re.UNICODE)


# Remove Vietnamese diacritics and normalise to ASCII for accent-insensitive comparison.
def _strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


# Tokenise Vietnamese text using pyvi when available, falling back to whitespace split.
# Every token has leading/trailing punctuation stripped (see _PUNCT_STRIP_RE)
# so a token glued to sentence punctuation ("An?", "công_ty,") still matches
# the same word appearing cleanly elsewhere in the corpus.
def _segment_vi(text: str) -> list[str]:
    text = (text or "").strip().lower()
    if not text:
        return []
    if ViTokenizer is not None:
        raw_tokens = [t.replace("_", " ") for t in ViTokenizer.tokenize(text).split()]
    else:
        raw_tokens = text.split()
    return [t for t in (_PUNCT_STRIP_RE.sub("", tok) for tok in raw_tokens) if t]


class ChromaRepository:
    _lock = threading.RLock()
    _client = None
    # Class-level collection cache shared across instances.
    _collections: dict[str, Any] = {}

    # Return the shared ChromaDB HTTP client, initialising it on first use.
    def _get_client(self) -> chromadb.HttpClient:
        if ChromaRepository._client is None:
            with ChromaRepository._lock:
                if ChromaRepository._client is None:
                    ChromaRepository._client = chromadb.HttpClient(
                        host=settings.chroma_host,
                        port=settings.chroma_port,
                    )
        return ChromaRepository._client

    # Return the named collection, creating it with cosine space if absent.
    def _get_collection(self, collection_name: str | None = None):
        name = collection_name or settings.chroma_collection
        if name not in ChromaRepository._collections:
            with ChromaRepository._lock:
                if name not in ChromaRepository._collections:
                    client = self._get_client()
                    ChromaRepository._collections[name] = client.get_or_create_collection(
                        name=name,
                        metadata={"hnsw:space": "cosine"},
                    )
        return ChromaRepository._collections[name]

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    # Flatten a metadata dict to scalar values; non-scalars are JSON-encoded.
    def _flatten_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for k, v in (metadata or {}).items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                flat[k] = v
            else:
                flat[k] = json.dumps(v, ensure_ascii=False)
        return flat

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    # Upsert a single chunk with its embedding and metadata.
    def upsert(
        self,
        *,
        chunk_id:      str,
        document_text: str,
        embedding:     list[float],
        metadata:      dict,
    ) -> None:
        collection = self._get_collection()
        collection.upsert(
            ids=[chunk_id],
            documents=[document_text],
            embeddings=[embedding],
            metadatas=[self._flatten_metadata(metadata)],
        )

    # Delete chunks by their IDs.
    def delete_by_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        self._get_collection().delete(ids=ids)

    # Return {chunk_id: metadata_dict} for the given IDs (present entries only).
    def get_metadatas_by_ids(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        raw = self._get_collection().get(ids=ids, include=["metadatas"])
        result = {}
        for chunk_id, meta in zip(raw.get("ids", []), raw.get("metadatas", [])):
            result[chunk_id] = meta or {}
        return result

    # Apply metadata_updates to all chunks matching chunk_ids via an upsert.
    def update_document_metadata(
        self, chunk_ids: list[str], metadata_updates: dict
    ) -> None:
        if not chunk_ids:
            return
        collection = self._get_collection()
        raw = collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas", "embeddings"],
        )
        ids        = raw.get("ids",        [])
        docs       = raw.get("documents",  [])
        metadatas  = raw.get("metadatas",  [])
        embeddings = raw.get("embeddings", [])
        if not ids:
            return

        new_metadatas = []
        for meta in metadatas:
            updated = dict(meta or {})
            for k, v in metadata_updates.items():
                updated[k] = "" if v is None else v
            new_metadatas.append(updated)

        collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=new_metadatas,
        )

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    # Query the collection by embedding vector; caps n_results to avoid HNSW errors.
    def query_by_embedding(
            self,
            *,
            embedding: list[float],
            top_k: int = 5,
            where: dict | None = None,
            collection_name: str | None = None,
        ) -> dict:
            collection = self._get_collection(collection_name)
            # Cap n_results to collection size to avoid HNSW "ef or M too small" error
            total = collection.count()
            n_results = min(top_k, max(1, total))
            kwargs: dict[str, Any] = dict(
                query_embeddings=[embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            if where:
                kwargs["where"] = where
            try:
                return collection.query(**kwargs)
            except RuntimeError:
                # Filtered subset smaller than n_results — retry with n_results=1
                kwargs["n_results"] = 1
                try:
                    return collection.query(**kwargs)
                except RuntimeError:
                    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    # ------------------------------------------------------------------
    # Keyword / lexical search
    # ------------------------------------------------------------------

    # Keyword search using where_document=$contains to avoid full-corpus scan;
    # scores candidates with token-overlap F1. Distances are 1 − f1_score.
    def query_by_keyword(
        self,
        *,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> dict:
        collection = self._get_collection()
        empty: dict = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query = (query or "").strip()
        if not query:
            return empty

        tokens = [t for t in _segment_vi(query) if len(t) > 1]
        if not tokens:
            return empty

        meaningful = [t for t in tokens if t not in _VN_STOPWORDS] or tokens
        primary_token = max(meaningful, key=len)

        # fetch only matching documents – much cheaper than full scan
        get_kwargs: dict[str, Any] = dict(
            where_document={"$contains": primary_token},
            include=["documents", "metadatas"],
        )
        if where:
            get_kwargs["where"] = where

        try:
            raw = collection.get(**get_kwargs)
        except Exception:
            # where_document may not be supported in older Chroma; degrade gracefully
            raw = collection.get(
                **({"where": where} if where else {}),
                include=["documents", "metadatas"],
            )

        ids       = raw.get("ids",       []) or []
        docs      = raw.get("documents", []) or []
        metadatas = raw.get("metadatas", []) or []

        if not ids:
            fallback_kwargs: dict[str, Any] = dict(include=["documents", "metadatas"])
            if where:
                fallback_kwargs["where"] = where
            try:
                raw = collection.get(**fallback_kwargs)
            except Exception:
                return empty
            ids       = raw.get("ids",       []) or []
            docs      = raw.get("documents", []) or []
            metadatas = raw.get("metadatas", []) or []

        if not ids:
            return empty

        q_set = {_strip_accents(t) for t in tokens}
        scored: list[tuple[float, int]] = []

        for i, doc in enumerate(docs):
            if not doc:
                continue
            d_tokens = [t for t in _segment_vi(str(doc)) if len(t) > 1]
            if not d_tokens:
                continue
            d_set = {_strip_accents(t) for t in d_tokens}
            overlap = len(q_set & d_set)
            if overlap == 0:
                continue
            precision = overlap / max(1, len(d_set))
            recall    = overlap / max(1, len(q_set))
            f1        = 2 * precision * recall / max(1e-9, precision + recall)
            scored.append((f1, i))

        if not scored:
            return empty

        scored.sort(key=lambda x: x[0], reverse=True)
        top_idx = [i for _, i in scored[:top_k]]

        return {
            "ids":       [[ids[i]       for i in top_idx]],
            "documents": [[docs[i]      for i in top_idx]],
            "metadatas": [[metadatas[i] for i in top_idx]],
            # Lexical "distance" = 1 − f1_score  (analogous to cosine distance)
            "distances": [[1.0 - scored[j][0] for j in range(len(top_idx))]],
        }

    # ------------------------------------------------------------------
    # Fetch raw candidates for external BM25 scoring
    # ------------------------------------------------------------------

    # Return the max chunk_sensitivity for all chunks of a document (from Chroma metadata).
    def get_max_chunk_sensitivity(self, document_id: str) -> int:
        try:
            raw = self._get_collection().get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            metadatas = raw.get("metadatas") or []
            max_sens = 1
            for m in metadatas:
                val = int((m or {}).get("chunk_sensitivity") or (m or {}).get("sensitivity") or 1)
                if val > max_sens:
                    max_sens = val
            return max_sens
        except Exception:
            return 1

    # Return raw documents and metadata for external BM25 re-ranking.
    def get_documents_for_bm25(
            self,
            *,
            where: dict | None = None,
            limit: int | None = None,
            collection_name: str | None = None,
        ) -> dict:
            collection = self._get_collection(collection_name)
            get_kwargs: dict[str, Any] = dict(include=["documents", "metadatas"])
            if where:
                get_kwargs["where"] = where
            if limit:
                get_kwargs["limit"] = limit
            try:
                raw = collection.get(**get_kwargs)
            except Exception:
                return {"ids": [], "documents": [], "metadatas": []}
            return {
                "ids": raw.get("ids", []) or [],
                "documents": raw.get("documents", []) or [],
                "metadatas": raw.get("metadatas", []) or [],
            }