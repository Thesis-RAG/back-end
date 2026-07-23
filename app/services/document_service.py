"""
Service for document CRUD, versioning, FGA synchronisation, and ingest job management.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.fga.adapter import fga_adapter
from app.models.chunk_embedding import ChunkEmbedding
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_version import DocumentVersion
from app.models.document_entity_action import DocumentEntityAction
from app.models.job import Job
from app.models.job_step import JobStep
from app.models.org_unit_instance import OrgUnitInstance
from app.models.policy_snapshot import DocumentPolicySnapshot
from app.models.user import User
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.version_repository import VersionRepository
from app.schemas.document import ChunkingConfig, DocumentCreateRequest, DocumentUpdateRequest
from app.services.audit_service import audit_service
from app.services.chroma_service import chroma_service
from app.services.entity_extractor import adjust_chunk_sensitivity
from app.services.job_service import job_service
from app.services.oui_tree_service import oui_tree_service
from app.services.sensitivity_levels import MIN_SENSITIVITY
from app.services.storage_service import storage_service
from app.services.user_service import user_service as _user_service
from app.services.entity_policy_service import normalize_actions

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self):
        self.docs = DocumentRepository()
        self.versions = VersionRepository()
        self.chunks = ChunkRepository()

    # ── Permission helpers ────────────────────────────────────────────────────

    # Return True if the user belongs to the corporate root OUI.
    def _is_corp_member(self, db: Session, user: User) -> bool:
        resp = _user_service.build_user_response(db, user)
        return resp.is_corp_member

    # Return the user's maximum clearance level.
    def _user_clearance(self, db: Session, user: User) -> int:
        resp = _user_service.build_user_response(db, user)
        return resp.max_clearance

    # Return True if FGA grants view access; owner always passes.
    def _can_view(self, db: Session, user: User, doc: Document) -> tuple[bool, str]:
        if doc.owner_user_id == user.id:
            return True, ""
        if fga_adapter.can_view(user.id, doc.id, self._user_clearance(db, user)):
            return True, ""
        return False, "No permission to view this document"

    # Return True if FGA grants edit access; owner always passes.
    def _can_edit(self, db: Session, user: User, doc: Document) -> tuple[bool, str]:
        if doc.owner_user_id == user.id:
            return True, ""
        if fga_adapter.can_edit(user.id, doc.id, self._user_clearance(db, user)):
            return True, ""
        return False, "No permission to edit this document"

    # ── FGA sync ──────────────────────────────────────────────────────────────

    # Delete and re-sync all FGA tuples for a document.
    def _sync_fga(self, db: Session, doc: Document) -> None:
        old = fga_adapter.get_document_tuples(doc.id)
        fga_adapter.delete_document_tuples(doc.id, old)
        fga_adapter.sync_document_tuples(db, doc)

    # ── Policy contract ───────────────────────────────────────────────────────

    # Persist a neutral snapshot for compatibility; runtime contracts are
    # generated from the version's per-entity actions during retrieval.
    def _entity_action_snapshot(self, doc: Document) -> dict:
        return {"mode": "entity_actions", "document_id": doc.id}

    # ── CRUD ──────────────────────────────────────────────────────────────────

    # Return all documents the user may view via FGA grants and ownership.
    def list_documents(self, db: Session, user: User) -> list[Document]:
        viewable_ids = fga_adapter.list_viewable_document_ids(user.id, self._user_clearance(db, user))
        public_filter = Document.sensitivity == MIN_SENSITIVITY
        if not viewable_ids:
            # Public documents are visible to every authenticated user, even
            # when they have no OUI grant. Owners can always view their own
            # documents as well.
            return db.query(Document).filter(
                or_(
                    public_filter,
                    Document.owner_user_id == user.id,
                )
            ).all()
        return db.query(Document).filter(
            or_(
                public_filter,
                Document.id.in_(viewable_ids),
                Document.owner_user_id == user.id,
            )
        ).all()

    # Fetch a document by ID, raising 403/404 if missing or inaccessible.
    def get_document(self, db: Session, user: User, doc_id: str) -> Document:
        doc = self.docs.get_by_id(db, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        if doc.sensitivity == MIN_SENSITIVITY:
            return doc
        ok, reason = self._can_view(db, user, doc)
        if not ok:
            raise HTTPException(status_code=403, detail=reason)
        return doc

    # Create a new document, validate OUI membership, persist it, and sync FGA tuples.
    def create_document(
        self, db: Session, user: User, payload: DocumentCreateRequest, trace_id: str
    ) -> Document:
        # Validate OUIs
        ouis = []
        user_oui_ids = {p.oui_id for p in user.oui_positions}
        is_corp = self._is_corp_member(db, user)

        for oui_id in payload.oui_ids:
            oui = db.get(OrgUnitInstance, oui_id)
            if not oui:
                raise HTTPException(status_code=404, detail=f"OUI not found: {oui_id}")

            if not is_corp:
                # User must belong to oui_id or one of its ancestors.

                ancestors = set(oui_tree_service.get_ancestors(db, oui_id))
                if not (user_oui_ids & ({oui_id} | ancestors)):
                    raise HTTPException(
                        status_code=403,
                        detail=f"No permission to assign document to OUI: {oui.name}",
                    )

            ouis.append(oui)

        doc = Document(
            title=payload.title,
            description=payload.description,
            owner_user_id=user.id,
            document_type=payload.document_type,
            sensitivity=payload.sensitivity,
            data_type=payload.data_type,
            tags=payload.tags or [],
            status="draft",
        )
        doc.ouis = ouis
        self.docs.create(db, doc)

        audit_service.log_action(
            db, trace_id=trace_id, user_id=user.id,
            action="document.create", resource_type="document",
            resource_id=doc.id, decision="allow",
            input_json=payload.model_dump(mode="json"),
        )
        db.commit()
        db.refresh(doc)
        self._sync_fga(db, doc)
        return doc

    # Update mutable document fields, re-sync chunk sensitivity on sensitivity change, and re-sync FGA.
    def update_document(
        self, db: Session, user: User, doc_id: str,
        payload: DocumentUpdateRequest, trace_id: str,
    ) -> Document:
        doc = self.docs.get_by_id(db, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        ok, reason = self._can_edit(db, user, doc)
        if not ok:
            raise HTTPException(status_code=403, detail=reason)

        fields_set = payload.model_fields_set

        if "oui_ids" in fields_set and payload.oui_ids is not None:
            ouis = []
            for oui_id in payload.oui_ids:
                oui = db.get(OrgUnitInstance, oui_id)
                if not oui:
                    raise HTTPException(status_code=404, detail=f"OUI not found: {oui_id}")
                ouis.append(oui)
            doc.ouis = ouis

        if payload.title is not None:
            doc.title = payload.title
        if payload.description is not None:
            doc.description = payload.description
        if payload.document_type is not None:
            doc.document_type = payload.document_type
        if payload.sensitivity is not None:
            old_sensitivity = doc.sensitivity
            doc.sensitivity = payload.sensitivity
            if old_sensitivity != payload.sensitivity:
                self._resync_chunk_sensitivity(db, doc, payload.sensitivity, old_sensitivity)
        if payload.tags is not None:
            doc.tags = payload.tags

        self.docs.save(db, doc)

        audit_service.log_action(
            db, trace_id=trace_id, user_id=user.id,
            action="document.update", resource_type="document",
            resource_id=doc.id, decision="allow",
            input_json=payload.model_dump(mode="json"),
        )
        db.commit()
        db.refresh(doc)
        self._sync_fga(db, doc)
        return doc

    # Re-compute chunk_sensitivity for all chunks after a doc sensitivity change.
    # Uses the old chunker/LLM sensitivity plus the persisted GLiNER entities.
    # Raises stay intact; reductions require the same strict common-public-
    # entity check used during ingest.
    def _resync_chunk_sensitivity(
        self, db: Session, doc: Document, new_sensitivity: int, old_sensitivity: int
    ) -> None:
        version_ids = [
            row[0]
            for row in db.query(DocumentVersion.id)
            .filter(DocumentVersion.document_id == doc.id)
            .all()
        ]
        if not version_ids:
            return

        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_version_id.in_(version_ids)
        ).all()
        if not chunks:
            return

        chunk_ids = [c.id for c in chunks]
        chunk_by_id = {c.id: c for c in chunks}

        try:
            chroma_metas = chroma_service.get_metadatas_by_ids(chunk_ids)
        except Exception as exc:
            logger.warning("Could not fetch Chroma metadata for doc %s: %s", doc.id, exc)
            chroma_metas = {}

        groups: dict[tuple[int, int], list[str]] = defaultdict(list)
        for cid in chunk_ids:
            chroma_meta = chroma_metas.get(cid, {})
            llm_cs = chroma_meta.get("llm_chunk_sensitivity")
            chunk = chunk_by_id[cid]
            meta = dict(chunk.metadata_json or {})
            if llm_cs is None:
                # Fallback for chunks ingested before llm_chunk_sensitivity
                # was stored: the old chunk metadata is the best available base.
                llm_cs = meta.get("llm_chunk_sensitivity", old_sensitivity)
            try:
                # Rebase the old chunker's relative delta to the new document
                # level before applying the GLiNER safety rule.
                rebased_llm_cs = int(new_sensitivity) + (int(llm_cs) - int(old_sensitivity))
            except (TypeError, ValueError):
                rebased_llm_cs = new_sensitivity
            rebased_llm_cs = max(1, min(5, rebased_llm_cs))
            new_cs = adjust_chunk_sensitivity(
                new_sensitivity,
                rebased_llm_cs,
                meta.get("detected_entities") or [],
                meta.get("sensitivity_flags") or set(),
            )
            groups[(new_cs, rebased_llm_cs)].append(cid)
            meta["chunk_sensitivity"] = new_cs
            meta["llm_chunk_sensitivity"] = rebased_llm_cs
            chunk.metadata_json = meta

        for (cs_val, llm_cs), cids in groups.items():
            try:
                chroma_service.update_document_metadata(
                    cids,
                    {
                        "chunk_sensitivity": cs_val,
                        "llm_chunk_sensitivity": llm_cs,
                        "sensitivity": new_sensitivity,
                    },
                )
            except Exception as exc:
                logger.warning("Failed to update Chroma chunk_sensitivity for doc %s: %s", doc.id, exc)

    # Return all versions of a document the user can view.
    def get_versions(self, db: Session, user: User, doc_id: str) -> list[DocumentVersion]:
        doc = self.get_document(db, user, doc_id)
        return self.versions.list_by_document(db, doc.id)

    # Upload a new file version, create storage/version/job records, and snapshot the policy contract.
    def create_version(
        self, db: Session, user: User, doc_id: str, *,
        raw_bytes: bytes, filename: str, content_type: str,
        trace_id: str, chunking_config: "ChunkingConfig | None" = None,
        entity_actions: list[dict] | None = None,
    ) -> tuple[Document, DocumentVersion, object, bool]:
        doc = self.docs.get_by_id(db, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        ok, reason = self._can_edit(db, user, doc)
        if not ok:
            raise HTTPException(status_code=403, detail=reason)

        if chunking_config is None:
            chunking_config = ChunkingConfig()

        last_no = self.docs.get_max_version_no(db, doc.id)
        version_no = last_no + 1
        checksum = storage_service.checksum(raw_bytes)
        object_key = f"documents/{doc.id}/versions/{version_no}/{Path(filename).name}"

        storage_obj = storage_service.upload_raw(
            db, data=raw_bytes, object_key=object_key,
            original_filename=Path(filename).name, content_type=content_type,
        )

        version = DocumentVersion(
            document_id=doc.id,
            version_no=version_no,
            file_name=Path(filename).name,
            mime_type=content_type,
            checksum=checksum,
            source_object_id=storage_obj.id,
            ingest_status="queued",
            parse_status="pending",
            chunk_status="pending",
            embed_status="pending",
            rule_version=settings.default_policy_version,
            chunk_config_json=chunking_config.to_json(),
        )
        self.versions.create(db, version)

        for sort_order, action in enumerate(normalize_actions(entity_actions or [])):
            db.add(DocumentEntityAction(
                document_version_id=version.id,
                entity_type=action["entity_type"],
                label=action.get("label"),
                action=action["action"],
                source=action.get("source", "manual"),
                enabled=action.get("enabled", True),
                detection_count=action.get("detection_count", 0),
                sort_order=sort_order,
                scope_oui_ids=action.get("scope_oui_ids") or [],
                scope_position_ids=action.get("scope_position_ids") or [],
                metadata_json=action.get("metadata_json") or {},
            ))

        doc.current_version_id = version.id
        doc.status = "uploaded"

        snapshot = DocumentPolicySnapshot(
            document_version_id=version.id,
            policy_version=settings.default_policy_version,
            contract_json=self._entity_action_snapshot(doc),
        )
        db.add(snapshot)

        job, created = job_service.create_or_get_ingest_job(
            db, trace_id=trace_id, document_id=doc.id,
            version_id=version.id, created_by_user_id=user.id,
            initial_status="pending_approval",
        )

        audit_service.log_action(
            db, trace_id=trace_id, user_id=user.id,
            action="document.version_upload", resource_type="document_version",
            resource_id=version.id, decision="allow",
            input_json={"filename": filename, "chunking_config": chunking_config.to_json()},
            output_json={"job_id": job.id, "version_no": version_no},
        )

        db.commit()
        db.refresh(doc)
        db.refresh(version)
        db.refresh(job)
        return doc, version, job, created

    # Create or retrieve an ingest job for a document version; optionally force a new job.
    def start_ingest(
        self, db: Session, user: User, doc_id: str, *,
        version_id: str | None, force_new: bool, trace_id: str,
    ):
        doc = self.docs.get_by_id(db, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        ok, reason = self._can_edit(db, user, doc)
        if not ok:
            raise HTTPException(status_code=403, detail=reason)

        if version_id:
            version = self.versions.get_by_id(db, version_id)
            if not version or version.document_id != doc.id:
                raise HTTPException(status_code=404, detail="Version not found")
        else:
            version = doc.current_version
            if not version:
                raise HTTPException(status_code=400, detail="No current version")

        job, created = job_service.create_or_get_ingest_job(
            db, trace_id=trace_id, document_id=doc.id,
            version_id=version.id, created_by_user_id=user.id, force_new=force_new,
        )

        db.commit()
        db.refresh(job)
        return job

    # Delete a document and all its associated versions, chunks, jobs, and FGA tuples.
    def delete_document(self, db: Session, user: User, doc_id: str, trace_id: str) -> None:
        doc = self.docs.get_by_id(db, doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Only corp members or the owner may delete the document.
        if doc.owner_user_id != user.id and not self._is_corp_member(db, user):
            raise HTTPException(status_code=403, detail="No permission to delete")

        old_tuples = fga_adapter.get_document_tuples(doc_id)
        fga_adapter.delete_document_tuples(doc_id, old_tuples)

        version_ids = [v.id for v in db.query(DocumentVersion).filter(
            DocumentVersion.document_id == doc_id).all()]

        if version_ids:
            chunk_ids = [c.id for c in db.query(DocumentChunk).filter(
                DocumentChunk.document_version_id.in_(version_ids)).all()]

            if chunk_ids:
                db.query(ChunkEmbedding).filter(
                    ChunkEmbedding.chunk_id.in_(chunk_ids)).delete(synchronize_session=False)
                db.query(DocumentChunk).filter(
                    DocumentChunk.document_version_id.in_(version_ids)).delete(synchronize_session=False)

            db.query(DocumentPolicySnapshot).filter(
                DocumentPolicySnapshot.document_version_id.in_(version_ids)).delete(synchronize_session=False)

            job_ids = [j.id for j in db.query(Job).filter(Job.document_id == doc_id).all()]
            if job_ids:
                db.query(JobStep).filter(JobStep.job_id.in_(job_ids)).delete(synchronize_session=False)
                db.query(Job).filter(Job.id.in_(job_ids)).delete(synchronize_session=False)

            doc.current_version_id = None
            db.flush()

            db.query(DocumentVersion).filter(
                DocumentVersion.document_id == doc_id).delete(synchronize_session=False)

            if chunk_ids:
                try:
                    chroma_service.delete_chunks(chunk_ids)
                except Exception as e:
                    logger.warning("Failed to delete Chroma chunks: %s", e)

        db.delete(doc)
        audit_service.log_action(
            db, trace_id=trace_id, user_id=user.id,
            action="document.delete", resource_type="document",
            resource_id=doc_id, decision="allow",
            input_json={"document_id": doc_id},
        )
        db.commit()


# Module-level singleton; imported by the document API router.
document_service = DocumentService()
