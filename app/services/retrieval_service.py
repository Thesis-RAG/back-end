"""
Retrieval service: hybrid RAG retrieval using Reciprocal Rank Fusion (RRF) over semantic and BM25 results,
with FGA permission gating and chunk-level sensitivity filtering.
"""
from __future__ import annotations
from rank_bm25 import BM25Okapi
from app.repositories.chroma_repository import ChromaRepository, _segment_vi

import logging
import math
from typing import Any
import re

from app.services.sensitivity_levels import SENSITIVITY_PATTERNS, MIN_SENSITIVITY, MAX_SENSITIVITY
from app.models.document import Document
from app.fga.adapter import fga_adapter
from app.repositories.chroma_repository import ChromaRepository
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

# RRF constant – higher k → smoother ranking (less sensitive to top ranks)
_RRF_K = 60
GMAIL_CHROMA_COLLECTION = "gmail_chunks"

# ----------------------------------------------------------------------
# Display-score combination (NOT used for ranking, only for UI %)
# ----------------------------------------------------------------------
# Ranking/sort still uses RRF (rank-based, robust, unaffected by this).
# These constants only control the absolute "match %" shown to the user.

# Weight given to semantic (cosine) score vs keyword (BM25) score
# when combining into a single display score.
_DISPLAY_ALPHA_SEMANTIC = 0.6

# BM25 raw-score saturation scale for the sigmoid below.
# Tune this based on the real BM25 raw-score distribution of your corpus
# (e.g. take the 90th percentile of raw scores from a sample of queries).
_BM25_SATURATION_SCALE = 5.0


# Convert a raw BM25 score to an absolute [0,1] relevance score via sigmoid saturation (batch-independent).
def _bm25_raw_to_absolute_score(raw_score: float) -> float:
    try:
        r = float(raw_score)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(r) or r <= 0.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-r / _BM25_SATURATION_SCALE))


class RetrievalService:
    def __init__(
        self,
        semantic_candidates: int = 20,   # how many to fetch from ANN
        lexical_candidates: int = 20,    # how many to fetch from keyword search
        minimum_score: float = 0.10,     # final RRF-normalised score threshold
    ):
        self.repo = ChromaRepository()
        self.semantic_candidates = semantic_candidates
        self.lexical_candidates  = lexical_candidates
        self.minimum_score       = minimum_score

    # ------------------------------------------------------------------
    # Similarity conversion  (cosine space only)
    # ------------------------------------------------------------------

    # Convert a Chroma cosine distance to similarity in [0, 1].
    def _cosine_sim(self, distance: Any) -> float:
        try:
            d = float(distance)
            if not math.isfinite(d):
                return 0.0
            # distance is already 1 - cosine_similarity, clamp for safety
            return max(0.0, min(1.0, 1.0 - d))
        except Exception:
            return 0.0
        
    # ------------------------------------------------------------------
    # BM25 lexical search 
    # ------------------------------------------------------------------

    # BM25 lexical search over the full Chroma corpus; returns top-k scored chunk dicts.
    def _bm25_search(
            self,
            *,
            query: str,
            top_k: int,
            where: dict | None = None,
            collection_name: str | None = None,
        ) -> list[dict]:
            raw = self.repo.get_documents_for_bm25(where=where, collection_name=collection_name)
            ids       = raw["ids"]
            docs      = raw["documents"]
            metadatas = raw["metadatas"]

            if not ids:
                return []

            tokenized_corpus = []
            valid_idx = []
            for i, doc in enumerate(docs):
                tokens = [t for t in _segment_vi(str(doc or "")) if len(t) > 1]
                if tokens:
                    tokenized_corpus.append(tokens)
                    valid_idx.append(i)

            if not tokenized_corpus:
                return []

            bm25 = BM25Okapi(tokenized_corpus)
            query_tokens = [t for t in _segment_vi(query) if len(t) > 1]
            if not query_tokens:
                return []

            scores = bm25.get_scores(query_tokens)
            scored = [(float(scores[j]), valid_idx[j]) for j in range(len(valid_idx))]
            scored = [s for s in scored if s[0] > 0.0]
            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:top_k]

            if not scored:
                return []

            # NOTE: `distance` here is only used by RRF ranking via _fuse(),
            # which only cares about rank order — not the absolute magnitude.
            # We still derive it from a batch-relative normalisation because
            # that's fine for *ranking* purposes. The user-facing keyword
            # score (see `_keyword_absolute_score` in registry items) is
            # computed separately from `_bm25_raw` using a batch-independent
            # saturation function, so the UI % is not affected by this.
            max_score = scored[0][0] or 1.0
            results = []
            for raw_score, idx in scored:
                norm_sim = raw_score / max_score
                distance = max(0.0, 1.0 - norm_sim)
                results.append({
                    "chunk_id":      ids[idx],
                    "document_text": docs[idx],
                    "metadata":      metadatas[idx] or {},
                    "distance":      distance,
                    "_bm25_raw":     raw_score,
                })
            return results

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    # Return the RRF score for a 0-based rank position.
    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (_RRF_K + rank + 1)

    # Merge semantic and lexical result lists via RRF; returns list sorted by rrf_score descending.
    def _fuse(
        self,
        semantic_results: list[dict],
        lexical_results: list[dict],
    ) -> list[dict]:
        scores: dict[str, float]  = {}
        registry: dict[str, dict] = {}

        def process(results: list[dict], source: str) -> None:
            for rank, item in enumerate(results):
                cid = item["chunk_id"]

                scores[cid] = scores.get(cid, 0.0) + self._rrf_score(rank)

                if cid not in registry:
                    registry[cid] = {
                        "chunk_id": item["chunk_id"],
                        "document_text": item["document_text"],
                        "metadata": item["metadata"],
                        "semantic_distance": None,
                        "lexical_distance": None,
                        "lexical_raw_score": None,
                        "sources": set(),
                    }

                if source == "semantic":
                    registry[cid]["semantic_distance"] = item.get("distance")

                if source == "lexical":
                    registry[cid]["lexical_distance"] = item.get("distance")
                    registry[cid]["lexical_raw_score"] = item.get("_bm25_raw")

                registry[cid]["sources"].add(source)

        process(semantic_results, "semantic")
        process(lexical_results,  "lexical")

        merged = []
        for cid, item in registry.items():
            rrf = scores[cid]
            merged.append({
                "chunk_id": item["chunk_id"],
                "document_text": item["document_text"],
                "metadata": item["metadata"],
                "semantic_distance": item.get("semantic_distance"),
                "lexical_distance": item.get("lexical_distance"),
                "lexical_raw_score": item.get("lexical_raw_score"),
                "rrf_score": round(rrf, 6),
                "sources": sorted(item["sources"]),
            })

        merged.sort(key=lambda x: x["rrf_score"], reverse=True)
        return merged

    # ------------------------------------------------------------------
    # Parse Chroma raw result → list[dict]
    # ------------------------------------------------------------------

    # Unpack a raw Chroma query result into a flat list of chunk dicts.
    def _parse_chroma(self, raw: dict) -> list[dict]:
        ids       = (raw.get("ids")       or [[]])[0]
        docs      = (raw.get("documents") or [[]])[0]
        metas     = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results = []
        for i, cid in enumerate(ids):
            results.append({
                "chunk_id":      cid,
                "document_text": docs[i]      if i < len(docs)      else "",
                "metadata":      metas[i]     if i < len(metas)     else {},
                "distance":      distances[i] if i < len(distances) else None,
            })
        return results

    # ------------------------------------------------------------------
    # Build where clause for Chroma
    # ------------------------------------------------------------------

    # Build a Chroma where-clause restricting results to allowed document IDs.
    def _build_where(
        self,
        allowed_doc_ids: list[str] | None,
        oui_ids: list[str] | None,
    ) -> dict | None:
        if allowed_doc_ids:
            return {"document_id": {"$in": allowed_doc_ids}}
        return None
    
    # Post-filter results to keep only chunks whose document belongs to at least one requested OUI.
    def _post_filter_oui(
        self,
        results: list[dict],
        oui_ids: list[str] | None,
    ) -> list[dict]:
        if not oui_ids:
            return results
        oui_set = set(oui_ids)
        filtered = []
        for chunk in results:
            # oui_id metadata is a comma-separated string, e.g. "abc,def,ghi".
            chunk_ouis = set(
                (chunk.get("metadata", {}).get("oui_id") or "").split(",")
            )
            chunk_ouis.discard("")
            if chunk_ouis & oui_set:  # intersection
                filtered.append(chunk)
        return filtered
    
    # Return chunk sensitivity as int 1-5; falls back to regex pattern scan if metadata is absent.
    def _classify_chunk_sensitivity(self, chunk: dict) -> int:
        meta = chunk.get("metadata", {}).get("sensitivity")
        if meta is not None:
            try:
                v = int(meta)
                if MIN_SENSITIVITY <= v <= MAX_SENSITIVITY:
                    return v
            except (ValueError, TypeError):
                pass

        # Fallback: regex scan on chunk text.
        text = chunk.get("document_text", "")
        for level in [5, 4, 3, 2]:  # scan from high to low
            for pattern in SENSITIVITY_PATTERNS.get(level, []):
                if re.search(pattern, text, re.IGNORECASE):
                    return level

        return 1  # default PUBLIC


    # Legacy sensitivity gate (superseded by _retrieve_main FGA logic; kept for compatibility).
    def _apply_sensitivity_gate(self, chunks: list[dict], user) -> list[dict]:
        if user is None:
            return [c for c in chunks
                    if self._classify_chunk_sensitivity(c) == SensitivityLevel.PUBLIC]

        max_clearance = getattr(user, "max_clearance", 1)
        # clearance 1-5 maps directly to SensitivityLevel 1-5.
        max_level = SensitivityLevel(max_clearance)
        allowed: list[dict] = []

        for chunk in chunks:
            chunk_level = self._classify_chunk_sensitivity(chunk)
            if chunk_level > max_level:
                logger.warning(
                    "SENSITIVITY GATE: blocked chunk=%s level=%s user=%s clearance=%d",
                    chunk.get("chunk_id"), chunk_level.name,
                    getattr(user, "id", "?"), max_clearance,
                )
                continue
            allowed.append(chunk)

        return allowed


    # Redact salary, phone, email, and ID numbers from a chunk for director-level access.
    def _redact_for_director(self, chunk: dict) -> dict:
        REDACT_PATTERNS = [
            (r"\b\d{1,3}(?:[.,]\d{3})+\s*(?:VND|đồng)\b", "[SỐ TIỀN ĐÃ ẨN]"),
            (r"\b(?:\+84|0)(?:3[2-9]|5[6-9]|7[0-9]|8[0-9]|9[0-9])[\s\-]?\d{3}[\s\-]?\d{3}\b",
            "[SĐT ĐÃ ẨN]"),
            (r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL ĐÃ ẨN]"),
            (r"\b\d{9}(?:\d{3})?\b", "[CCCD ĐÃ ẨN]"),
        ]
        new_chunk = dict(chunk)
        text = new_chunk.get("document_text", "")
        for pattern, replacement in REDACT_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        new_chunk["document_text"]    = text
        new_chunk["_director_redacted"] = True
        return new_chunk

    # ------------------------------------------------------------------
    # Absolute (batch-independent) display score
    # ------------------------------------------------------------------

    # Combine semantic similarity and BM25 score into an absolute [0,1] display score (batch-independent).
    def _compute_display_score(
        self,
        sem: float | None,
        lexical_raw_score: float | None,
    ) -> tuple[float, float | None]:
        sem_val = sem if sem is not None else 0.0
        kw_abs = (
            _bm25_raw_to_absolute_score(lexical_raw_score)
            if lexical_raw_score is not None
            else None
        )
        kw_val = kw_abs if kw_abs is not None else 0.0

        if sem is None and kw_abs is None:
            return 0.0, None

        # Single-source mode: use that score directly
        if sem is None:
            return round(min(1.0, max(0.0, kw_val)), 6), kw_abs
        if kw_abs is None:
            return round(min(1.0, max(0.0, sem_val)), 6), kw_abs

        # Hybrid: weighted combination
        display = (
            _DISPLAY_ALPHA_SEMANTIC * sem_val
            + (1.0 - _DISPLAY_ALPHA_SEMANTIC) * kw_val
        )
        return round(min(1.0, max(0.0, display)), 6), kw_abs

    # ------------------------------------------------------------------
    # Main retrieve
    # ------------------------------------------------------------------

    # Route the query to the appropriate retrieval method based on chat_mode.
    def retrieve(
        self,
        *,
        query: str,
        user=None,
        top_k: int = 5,
        mode: str = "hybrid",
        oui_ids: list[str] | None = None,
        chat_mode: str = "rag",
        db=None,
    ) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []

        if not embedding_service.is_configured():
            raise RuntimeError("Embedding service is not configured")

        # Resolve user_id early to avoid DetachedInstanceError.
        user_id = str(user.id) if user else None

        if chat_mode == "gmail":
            return self._retrieve_from_collection(
                query=query, user=user, top_k=top_k,
                collection_name=GMAIL_CHROMA_COLLECTION,
                extra_where={"user_id": {"$eq": user_id}} if user_id else None,
                semantic_candidates=40,
                lexical_candidates=40,
            )
        elif chat_mode == "all":
            rag_results = self._retrieve_main(query=query, user=user, top_k=top_k, oui_ids=oui_ids, db=db)
            gmail_results = self._retrieve_from_collection(
                query=query, user=user, top_k=top_k,
                collection_name=GMAIL_CHROMA_COLLECTION,
                extra_where={"user_id": {"$eq": user_id}} if user_id else None,
            )
            merged = rag_results + gmail_results
            merged.sort(key=lambda x: x.get("score", 0), reverse=True)
            return merged[:top_k]
        else:
            return self._retrieve_main(
                query=query, user=user, top_k=top_k, mode=mode, oui_ids=oui_ids, db=db
            )
            
            
    # Retrieve from an arbitrary Chroma collection (e.g. gmail_chunks) filtered only by extra_where.
    def _retrieve_from_collection(
        self,
        *,
        query: str,
        user=None,
        top_k: int = 5,
        collection_name: str,
        extra_where: dict | None = None,
        mode: str = "hybrid",
        semantic_candidates: int | None = None,
        lexical_candidates: int | None = None,
    ) -> list[dict]:
        where = extra_where
        sem_k = semantic_candidates or self.semantic_candidates
        lex_k = lexical_candidates or self.lexical_candidates

        semantic_list: list[dict] = []
        lexical_list: list[dict] = []

        if mode in ("semantic", "hybrid"):
            emb = embedding_service.embed(query)
            raw = self.repo.query_by_embedding(
                embedding=emb, top_k=sem_k, where=where,
                collection_name=collection_name,
            )
            semantic_list = self._parse_chroma(raw)

        if mode in ("keyword", "hybrid"):
            lexical_list = self._bm25_search(
                query=query, top_k=lex_k, where=where,
                collection_name=collection_name,
            )
            
        if not semantic_list and not lexical_list:
            return []

        fused = self._fuse(semantic_list, lexical_list) if mode == "hybrid" else (
            sorted([{**i, "rrf_score": self._cosine_sim(i.get("distance")), "sources": ["semantic"],
                     "semantic_distance": i.get("distance")}
                    for i in semantic_list], key=lambda x: x["rrf_score"], reverse=True)
            if mode == "semantic" else
            sorted([{**i, "rrf_score": max(0.0, 1.0 - float(i.get("distance", 1.0))), "sources": ["lexical"],
                     "lexical_raw_score": i.get("_bm25_raw")}
                    for i in lexical_list], key=lambda x: x["rrf_score"], reverse=True)
        )

        # NOTE: RRF score (rrf_score) is only used to ORDER `fused` (already
        # sorted above). It is intentionally NOT used to compute the
        # user-facing `score` below — that comes from an absolute,
        # batch-independent combination of semantic + keyword scores.
        results = []
        for item in fused:
            sem = self._cosine_sim(item.get("semantic_distance")) if item.get("semantic_distance") is not None else None
            lexical_raw = item.get("lexical_raw_score")
            display_score, kw_abs = self._compute_display_score(sem, lexical_raw)
            if display_score < self.minimum_score:
                continue
            results.append({
                "chunk_id": item["chunk_id"],
                "document_text": item["document_text"],
                "metadata": item["metadata"] or {},
                "score": display_score,
                "semantic_score": round(sem, 6) if sem is not None else None,
                "keyword_score": round(kw_abs, 6) if kw_abs is not None else None,
                "sources": item.get("sources", []),
            })

        return results[:top_k]
    

    # Retrieve from the main document_chunks collection with FGA gating and clearance-based blurring.
    def _retrieve_main(
        self,
        *,
        query: str,
        user=None,
        top_k: int = 5,
        mode: str = "hybrid",
        oui_ids: list[str] | None = None,
        db=None,
    ) -> list[dict]:
        if user is None:
            return []

        # ── User clearance ──────────────────────────────────────────────
        user_clearance = 1
        for uop in getattr(user, "oui_positions", []):
            if uop.position and uop.position.clearance > user_clearance:
                user_clearance = uop.position.clearance

        # ── FGA-allowed doc IDs ─────────────────────────────────────────
        fga_allowed_ids: set[str] = set(
            fga_adapter.list_viewable_document_ids(str(user.id), user_clearance)
        )

        # Public documents are visible to every authenticated user, including
        # documents without an OUI grant. Include them in the same allowed set
        # so retrieval does not label their chunks as permission-denied.
        if db is not None:
            public_rows = db.query(Document.id).filter(
                Document.sensitivity == MIN_SENSITIVITY
            ).all()
            fga_allowed_ids.update(str(row[0]) for row in public_rows)

        # ── Approved access requests (per-doc, not expired) ─────────────
        approved_doc_ids: set[str] = set()
        scope_mode = "full_db"
        if db is not None:
            from app.repositories.document_access_request_repository import doc_access_request_repo
            from app.repositories.system_setting_repository import system_setting_repository
            approved_doc_ids = doc_access_request_repo.get_active_approved_doc_ids(db, str(user.id))
            scope_mode = system_setting_repository.get(db, "query_scope_mode") or "full_db"

        # ── Chroma query scope ──────────────────────────────────────────
        # full_db: query all Chroma (no doc_id filter).
        # branch_only: query only docs in the user's OUI branch.
        if scope_mode == "branch_only" and db is not None:
            from app.services.oui_tree_service import oui_tree_service
            branch_oui_ids = oui_tree_service.get_user_branch_oui_ids(db, user)
            branch_doc_ids = oui_tree_service.get_doc_ids_for_oui_ids(db, branch_oui_ids)
            query_doc_ids = list(fga_allowed_ids | branch_doc_ids)
            where = self._build_where(query_doc_ids or None, oui_ids)
            if not query_doc_ids:
                return []
        else:
            # full_db: no filter — Chroma returns all.
            where = None if not oui_ids else self._build_where(None, oui_ids)

        semantic_list: list[dict] = []
        lexical_list: list[dict] = []

        if mode in ("semantic", "hybrid"):
            emb = embedding_service.embed(query)
            raw = self.repo.query_by_embedding(
                embedding=emb, top_k=self.semantic_candidates, where=where,
            )
            semantic_list = self._parse_chroma(raw)
            
            # print("========== TOP SEMANTIC ==========")
            # logger.info("========== TOP SEMANTIC ==========")

            for idx, r in enumerate(semantic_list[:5], start=1):
                md = r.get("metadata", {})

                # print(
                #     "[SEM %d] distance=%.6f heading=%s chunk=%s",
                #     idx,
                #     float(r.get("distance") or 999),
                #     md.get("section_heading"),
                #     r.get("chunk_id"),
                # )

                # print(
                #     "[SEM %d TEXT] %s",
                #     idx,
                #     (r.get("document_text") or ""),
                # )

        if mode in ("keyword", "hybrid"):
            lexical_list = self._bm25_search(
                query=query, top_k=self.lexical_candidates, where=where,
            )
            
            # print("========== TOP LEXICAL ==========")
            # logger.info("========== TOP LEXICAL ==========")

            for idx, r in enumerate(lexical_list[:5], start=1):
                md = r.get("metadata", {})

                # print(
                #     "[LEX %d] distance=%.6f heading=%s chunk=%s",
                #     idx,
                #     float(r.get("distance") or 999),
                #     md.get("section_heading"),
                #     r.get("chunk_id"),
                # )

                # print(
                #     "[LEX %d TEXT] %s",
                #     idx,
                #     (r.get("document_text") or "")[:300],
                # )

        if not semantic_list and not lexical_list:
            return []

        fused = self._fuse(semantic_list, lexical_list) if mode == "hybrid" else (
            sorted([{**i, "rrf_score": self._cosine_sim(i.get("distance")), "sources": ["semantic"],
                     "semantic_distance": i.get("distance")}
                    for i in semantic_list], key=lambda x: x["rrf_score"], reverse=True)
            if mode == "semantic" else
            sorted([{**i, "rrf_score": max(0.0, 1.0 - float(i.get("distance", 1.0))), "sources": ["lexical"],
                     "lexical_raw_score": i.get("_bm25_raw")}
                    for i in lexical_list], key=lambda x: x["rrf_score"], reverse=True)
        )

        # print("========== TOP FUSED ==========")
        # logger.info("========== TOP FUSED ==========")

        for idx, r in enumerate(fused[:10], start=1):
            md = r.get("metadata", {})

            # print(
            #     "[FUSED %d] rrf=%.6f heading=%s sources=%s",
            #     idx,
            #     r.get("rrf_score"),
            #     md.get("section_heading"),
            #     r.get("sources"),
            # )

        # NOTE: RRF score (rrf_score) is only used to ORDER `fused` (already
        # sorted above). It is intentionally NOT used to compute the
        # user-facing `score` below — that comes from an absolute,
        # batch-independent combination of semantic + keyword scores.
        # This is what fixes the "always one source at 100%" issue: the
        # top-ranked chunk no longer automatically gets score = 1.0 just
        # because it happened to rank first within THIS query's batch.
        results = []
        # FGA-allowed docs with chunks that exceed clearance and have no approval yet.
        # Pass 2 sets doc_restricted=True on all chunks of these docs so the frontend hides Eye/Plus.
        fga_has_blocked: set[str] = set()

        for item in fused:
            sem = self._cosine_sim(item.get("semantic_distance")) if item.get("semantic_distance") is not None else None
            lexical_raw = item.get("lexical_raw_score")
            display_score, kw_abs = self._compute_display_score(sem, lexical_raw)
            if display_score < self.minimum_score:
                continue

            meta     = item.get("metadata") or {}
            doc_id   = meta.get("document_id", "")
            chunk_sens = int(meta.get("chunk_sensitivity") or meta.get("sensitivity") or 1)

            fga_allow    = doc_id in fga_allowed_ids
            has_approval = doc_id in approved_doc_ids

            if fga_allow:
                # FGA allow → return ALL chunks; blur those exceeding clearance without approval.
                # chunk_blurred=True if chunk_sens > clearance and not yet approved.
                is_blurred = chunk_sens > user_clearance and not has_approval
                if is_blurred:
                    fga_has_blocked.add(doc_id)
                results.append({
                    "chunk_id":       item["chunk_id"],
                    "document_text":  item["document_text"],
                    "metadata":       meta,
                    "score":          display_score,
                    "semantic_score": round(sem, 6) if sem is not None else None,
                    "keyword_score":  round(kw_abs, 6) if kw_abs is not None else None,
                    "sources":        item.get("sources", []),
                    "doc_restricted": False,
                    "chunk_blurred":  is_blurred,
                })
            else:
                # FGA deny — include only if chunk_sensitivity ≤ clearance.
                if chunk_sens <= user_clearance:
                    results.append({
                        "chunk_id":       item["chunk_id"],
                        "document_text":  item["document_text"],
                        "metadata":       meta,
                        "score":          display_score,
                        "semantic_score": round(sem, 6) if sem is not None else None,
                        "keyword_score":  round(kw_abs, 6) if kw_abs is not None else None,
                        "sources":        item.get("sources", []),
                        "doc_restricted": True,   # frontend hides Eye/Plus
                        "chunk_blurred":  False,
                    })

        # Pass 2: FGA-allowed docs with blocked chunks → set doc_restricted=True on all their chunks.
        if fga_has_blocked:
            for r in results:
                if (r.get("metadata") or {}).get("document_id") in fga_has_blocked:
                    r["doc_restricted"] = True

        results = self._post_filter_oui(results, oui_ids)
        return results[:top_k]


    # Return the highest clearance level across all of the user's OUI positions (1-5).
    def _get_user_clearance(self, user) -> int:
        if user is None:
            return 1

        oui_positions = getattr(user, "oui_positions", None) or []
        max_clearance = 1

        for oui_pos in oui_positions:
            position = getattr(oui_pos, "position", None)
            if position is None:
                continue
            c = getattr(position, "clearance", 1)
            if c > max_clearance:
                max_clearance = c

        return max_clearance


    # Filter chunks whose sensitivity exceeds the user's maximum clearance level.
    def _apply_sensitivity_gate(self, chunks: list[dict], user) -> list[dict]:
        user_clearance = self._get_user_clearance(user)
        allowed = []

        for chunk in chunks:
            chunk_sensitivity = self._classify_chunk_sensitivity(chunk)
            if chunk_sensitivity > user_clearance:
                logger.warning(
                    "SENSITIVITY GATE: blocked chunk=%s sensitivity=%d user=%s clearance=%d",
                    chunk.get("chunk_id"), chunk_sensitivity,
                    getattr(user, "id", "?"), user_clearance,
                )
                continue
            allowed.append(chunk)

        return allowed
    


# Module-level singleton; imported by the chat service.
retrieval_service = RetrievalService(
    semantic_candidates=20,
    lexical_candidates=20,
    minimum_score=0.0,
)
