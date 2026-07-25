"""
Ingest pipeline service: orchestrates download → parse → chunk → entity detection → embed for a document version.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.chunk_embedding import ChunkEmbedding
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_version import DocumentVersion
from app.models.document_entity_action import DocumentEntityAction
from app.models.policy_snapshot import DocumentPolicySnapshot
from app.schemas.document import ChunkingConfig
from app.services.audit_service import audit_service
from app.services.chunker_service import ChunkConfig, chunker_service
from app.services.chroma_service import chroma_service
from app.services.embedding_service import embedding_service
from app.services.job_service import job_service
from app.services.parser_service import parser_service
from app.services.storage_service import storage_service
from app.services.entity_policy_service import entity_policy_service

logger = logging.getLogger(__name__)


class IngestPipelineService:

    # Execute the full ingest pipeline for a job, writing step records and updating status fields.
    def run(self, db: Session, job_id: str):
        job = job_service.get_job(db, job_id)
        if not job:
            return
        if job.status == "succeeded":
            return

        start_total = time.perf_counter()
        job.status   = "running"
        job.progress = 1
        db.commit()

        doc     = db.get(Document,        job.document_id)
        version = db.get(DocumentVersion, job.document_version_id)
        if not doc or not version:
            job.status        = "failed"
            job.error_message = "Document/version missing"
            db.commit()
            return

        try:
            pre_ingest_status     = doc.status   # saved to restore doc status after ingest.
            doc.status            = "processing"
            version.ingest_status = "running"
            version.parse_status  = "running"
            version.chunk_status  = "pending"
            version.embed_status  = "pending"
            db.commit()

            # ── Read chunking config from version ─────────────────────
            # version.chunk_config_json is saved at upload time.
            # If None (legacy version) → use legacy defaults.
            chunking_cfg = ChunkConfig(
                **ChunkingConfig.from_json(version.chunk_config_json).model_dump(
                    exclude={"embed_model"}  # ChunkingConfig không có field này
                ),
                embed_model="sentence-transformers/all-MiniLM-L6-v2",
            )
            logger.info(
                "Ingest job=%s mode=%s max_tokens=%d overlap=%d ocr=%s",
                job_id, chunking_cfg.mode, chunking_cfg.max_tokens,
                chunking_cfg.overlap_tokens, chunking_cfg.ocr,
            )

            # ── clean re-run ──────────────────────────────────────────
            db.query(DocumentChunk).filter(
                DocumentChunk.document_version_id == version.id
            ).delete(synchronize_session=False)
            db.commit()

            # ── download ──────────────────────────────────────────────
            raw_step = job_service.add_step(
                db, job_id=job.id, step_name="download_raw",
                detail_json={
                    "bucket":     version.source_object.bucket,
                    "object_key": version.source_object.object_key,
                },
            )
            t0        = time.perf_counter()
            raw_bytes = storage_service.download(
                version.source_object.bucket, version.source_object.object_key
            )
            job_service.finish_step(
                db, raw_step,
                detail_json={
                    "size_bytes": len(raw_bytes),
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                },
            )
            db.commit()

            # ── parse ─────────────────────────────────────────────────
            parse_step = job_service.add_step(
                db, job_id=job.id, step_name="parse",
                detail_json={"filename": version.file_name},
            )
            t0     = time.perf_counter()
            parsed = parser_service.parse(raw_bytes, version.file_name, version.mime_type)
            job_service.finish_step(
                db, parse_step,
                detail_json={
                    "pages":      len(parsed.pages),
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                },
            )
            version.parse_status = "completed"
            db.commit()

            normalized_key = (
                f"documents/{doc.id}/versions/{version.version_no}/normalized.txt"
            )
            normalized_obj = storage_service.upload_processed_text(
                db,
                text=parsed.full_text,
                object_key=normalized_key,
                original_filename=f"{Path(version.file_name).stem}.normalized.txt",
            )
            version.normalized_object_id = normalized_obj.id
            db.commit()

            # Detect the complete document once and retain the snapshot for
            # the upload editor and the direct-file entity access gate.
            configured_actions = db.query(DocumentEntityAction).filter(
                DocumentEntityAction.document_version_id == version.id,
                DocumentEntityAction.enabled.is_(True),
            ).all()
            configured_by_type = {row.entity_type: row for row in configured_actions}
            full_detection = entity_policy_service.detect_full_text(parsed.full_text, db)
            confirmed_labels = sorted(
                set(full_detection["entity_types"]) & set(configured_by_type)
            )
            version.entity_detection_json = {
                "entity_types": confirmed_labels,
                "confirmed_labels": confirmed_labels,
                "entities": full_detection["entities"],
            }
            for entity_type, row in configured_by_type.items():
                row.detection_count = sum(
                    1 for entity in full_detection["entities"]
                    if entity.get("label") == entity_type
                )
            version.confirmed_labels_json = confirmed_labels
            snapshot = db.query(DocumentPolicySnapshot).filter(
                DocumentPolicySnapshot.document_version_id == version.id,
            ).first()
            if snapshot:
                snapshot.confirmed_labels_json = confirmed_labels
                contract = dict(snapshot.contract_json or {})
                contract["confirmed_labels"] = confirmed_labels
                snapshot.contract_json = contract
            db.commit()

            policy_action_summary = {"block": 0, "mask": 0, "full": 0}
            for row in configured_by_type.values():
                if row.entity_type in confirmed_labels:
                    policy_action_summary[row.action] = policy_action_summary.get(row.action, 0) + 1
            audit_service.log_action(
                db,
                trace_id=job.trace_id,
                user_id=job.created_by_user_id,
                action="policy.document_rules_applied",
                resource_type="document_version",
                resource_id=version.id,
                decision="allow",
                job_id=job.id,
                output_json={
                    "policy_profile": version.policy_profile,
                    "policy_version": version.policy_version,
                    "confirmed_labels": confirmed_labels,
                    "action_summary": policy_action_summary,
                },
            )
            db.commit()

            # ── chunk ─────────────────────────────────────────────────
            chunk_step = job_service.add_step(
                db, job_id=job.id, step_name="chunk",
                detail_json={"mode": chunking_cfg.mode},
            )
            t0 = time.perf_counter()

            # Pass the per-version chunking config to the chunker.
            chunks = chunker_service.chunk(parsed, config=chunking_cfg, doc_sensitivity=doc.sensitivity)

            from app.services.entity_extractor import (
                compute_chunk_sensitivity,
                extract_realtime_batch_detailed,
            )
            chunk_details = extract_realtime_batch_detailed(
                [c.get("chunk_text") or "" for c in chunks],
                db=db,
                labels=confirmed_labels,
            )
            for chunk_dict, detail in zip(chunks, chunk_details):
                metadata_json = chunk_dict.setdefault("metadata_json", {})
                entity_types = sorted({
                    str(value) for value in (detail.get("entity_types") or [])
                } & set(confirmed_labels))
                # Frozen snapshot of each detected type's (action, sensitivity)
                # at ingest time — the sole input `_resync_chunk_sensitivity`
                # later re-reads to recompute chunk_sensitivity without
                # re-running entity detection.
                entity_snapshot = {
                    label: {
                        "action": configured_by_type[label].action,
                        "sensitivity": int(configured_by_type[label].sensitivity or 1),
                    }
                    for label in entity_types
                    if label in configured_by_type
                }
                metadata_json["entity_types"] = entity_types
                metadata_json["detected_entities"] = detail.get("entities") or []
                metadata_json["sensitivity_flags"] = sorted(detail.get("flags") or [])
                metadata_json["confirmed_labels"] = confirmed_labels
                metadata_json["label_set_version"] = version.policy_version
                metadata_json["entity_actions"] = {
                    label: info["action"] for label, info in entity_snapshot.items()
                }
                metadata_json["entity_policy_snapshot"] = entity_snapshot
                metadata_json["chunk_sensitivity"] = compute_chunk_sensitivity(
                    doc.sensitivity, entity_snapshot
                )

            # Ontology graph: record which entity types actually showed up in
            # a document of this type — best-effort, never blocks ingest.
            all_detected_types = {
                t for chunk_dict in chunks
                for t in (chunk_dict.get("metadata_json") or {}).get("entity_types") or []
            }
            if all_detected_types:
                from app.services.ontology_sync_service import ontology_sync_service
                from app.services.policy_rule_service import policy_rule_service as _policy_rule_service

                for rule in _policy_rule_service.rules_by_key(db, all_detected_types).values():
                    ontology_sync_service.sync_document_entity(doc.document_type, rule.id)

            chunk_models: list[DocumentChunk] = []
            for c in chunks:
                chunk = DocumentChunk(
                    document_version_id=version.id,
                    chunk_index=c["chunk_index"],
                    chunk_text=c["chunk_text"],
                    page_start=c["page_start"],
                    page_end=c["page_end"],
                    token_count=c["token_count"],
                    metadata_json=c["metadata_json"],
                    chunk_hash=c["chunk_hash"],
                )
                db.add(chunk)
                db.flush()
                chunk_models.append(chunk)

            job_service.finish_step(
                db, chunk_step,
                detail_json={
                    "chunk_count": len(chunk_models),
                    "mode":        chunking_cfg.mode,
                    "latency_ms":  int((time.perf_counter() - t0) * 1000),
                },
            )
            version.chunk_status = "completed"
            db.commit()

            db.commit()

            # ── embed  (BATCH) ────────────────────────────────────────
            embed_step = job_service.add_step(
                db, job_id=job.id, step_name="embed",
                detail_json={"dimensions": embedding_service.dimensions},
            )
            t0 = time.perf_counter()

            embed_texts = [c.get("embed_text") or c["chunk_text"] for c in chunks]
            vectors     = embedding_service.embed_many(embed_texts)

            for chunk_model, chunk_dict, vector in zip(chunk_models, chunks, vectors):
                meta_json = chunk_dict.get("metadata_json") or {}
                metadata = {
                    "document_id":         doc.id,
                    "document_version_id": version.id,
                    "document_title":      doc.title,
                    "oui_ids":             ",".join(sorted([o.id for o in (doc.ouis or [])])),
                    "document_type":       doc.document_type,
                    "sensitivity":         doc.sensitivity,
                    "data_type":           doc.data_type,
                    "chunk_index":         chunk_model.chunk_index,
                    "page_start":          chunk_model.page_start,
                    "page_end":            chunk_model.page_end,
                    "chunk_hash":          chunk_model.chunk_hash,
                    "section_heading":     meta_json.get("section_heading", ""),
                    "section_index":       meta_json.get("section_index", 0),
                    "position_ratio":      meta_json.get("position_ratio", 0.0),
                    "chunker_mode":        meta_json.get("chunker_mode", "legacy"),
                    "chunk_sensitivity":     meta_json.get("chunk_sensitivity", doc.sensitivity),
                    "entity_types": ",".join(meta_json.get("entity_types") or []),
                    "confirmed_labels": ",".join(meta_json.get("confirmed_labels") or []),
                    "label_set_version": meta_json.get("label_set_version", version.policy_version),
                    "policy_profile": version.policy_profile,
                    "policy_version": version.policy_version,
                }
                chroma_service.upsert_chunk(
                    chunk_id=chunk_model.id,
                        document_text=chunk_dict.get("embed_text") or chunk_dict["chunk_text"],  
                    embedding=vector,
                    metadata=metadata,
                )
                db.add(
                    ChunkEmbedding(
                        chunk_id=chunk_model.id,
                        vector_db="chroma",
                        collection_name=settings.chroma_collection,
                        vector_id=chunk_model.id,
                        embedding_model=embedding_service.model_name,
                        dimensions=embedding_service.dimensions,
                        embedding_status="completed",
                    )
                )

            job_service.finish_step(
                db, embed_step,
                detail_json={
                    "embedded_chunks": len(chunk_models),
                    "latency_ms":      int((time.perf_counter() - t0) * 1000),
                },
            )
            version.embed_status  = "completed"
            version.ingest_status = "succeeded"

            if pre_ingest_status == "approved":
                doc.status = "approved"
            else:
                creator = job.created_by
                if creator:
                    from app.services.user_service import user_service as _user_service
                    from app.db.session import SessionLocal
                    with SessionLocal() as tmp_db:
                        creator_resp = _user_service.build_user_response(tmp_db, creator)
                        doc.status = "approved" if creator_resp.is_corp_member else "review"
                else:
                    doc.status = "review"

            doc.current_version_id = version.id

            audit_service.emit_event(
                db,
                event_type="document.embedded",
                aggregate_type="document_version",
                aggregate_id=version.id,
                payload_json={
                    "document_id":         doc.id,
                    "document_version_id": version.id,
                    "chunk_count":         len(chunk_models),
                    "collection":          settings.chroma_collection,
                    "chunker_mode":        chunking_cfg.mode,
                },
            )
            audit_service.log_action(
                db,
                trace_id=job.trace_id,
                user_id=job.created_by_user_id,
                action="document.ingest.completed",
                resource_type="document_version",
                resource_id=version.id,
                decision="allow",
                job_id=job.id,
                output_json={
                    "chunk_count":     len(chunk_models),
                    "embedding_model": embedding_service.model_name,
                    "chunker_mode":    chunking_cfg.mode,
                },
                latency_ms=int((time.perf_counter() - start_total) * 1000),
            )

            job.status      = "succeeded"
            job.progress    = 100
            job.finished_at = datetime.now(timezone.utc).isoformat()
            db.commit()

        except Exception as exc:
            db.rollback()
            job     = job_service.get_job(db, job.id)
            doc     = db.get(Document,        job.document_id)         if job else None
            version = db.get(DocumentVersion, job.document_version_id) if job else None

            if job:
                job.status        = "failed"
                job.error_message = str(exc)
                job.progress      = min(job.progress or 0, 99)
                job.finished_at   = datetime.now(timezone.utc).isoformat()
            if doc:
                doc.status = "failed"
            if version:
                version.ingest_status = "failed"
                version.error_message = str(exc)
                version.parse_status  = version.parse_status or "failed"
                version.chunk_status  = "failed"
                version.embed_status  = "failed"

            job_service.add_step(
                db,
                job_id=job.id if job else job_id,
                step_name="failed",
                detail_json={"error": str(exc)},
            )
            audit_service.log_action(
                db,
                trace_id=job.trace_id         if job     else "unknown",
                user_id=job.created_by_user_id if job     else None,
                action="document.ingest.failed",
                resource_type="document_version",
                resource_id=version.id         if version else None,
                decision="deny",
                job_id=job.id                  if job     else None,
                output_json={"error": str(exc)},
            )
            db.commit()
            raise


# Module-level singleton; imported by the ingest worker and job API router.
ingest_pipeline_service = IngestPipelineService()
