"""
FastAPI dependency providers: database session, trace ID, and authenticated user.
"""
import uuid

from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session, joinedload, selectinload

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.user import User
from app.models.user_oui_position import UserOuiPosition
from app.models.position import Position
from app.models.org_unit_instance import OrgUnitInstance

# Bearer token extractor; points to the login endpoint as the token URL.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# Return the trace ID attached to the request state, or generate a new one.
def get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", uuid.uuid4().hex)


# Decode the Bearer token and return the fully loaded User with OUI positions.
def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    # user_service.build_user_response() (and anything else that walks
    # oui_positions -> oui -> ou / .parents) may run against a DIFFERENT,
    # later-opened DB session than this one — eg chat's streaming pipeline
    # loads `user` here via get_current_user's own session, then reuses that
    # same User object from inside a freshly opened stream_db session several
    # calls deep (retrieval_service._retrieve_main -> document_service
    # .visible_document_ids -> _user_clearance -> build_user_response). Any
    # relationship NOT eagerly loaded right here would try to lazy-load
    # against whichever session touches it next, which raises
    # DetachedInstanceError once this dependency's own session is closed —
    # so every attribute build_user_response reads off oui_positions must be
    # eager-loaded up front, not just .position.
    user = (
        db.query(User)
        .options(
            joinedload(User.oui_positions).joinedload(UserOuiPosition.position),
            joinedload(User.oui_positions).joinedload(UserOuiPosition.oui).joinedload(OrgUnitInstance.ou),
            joinedload(User.oui_positions).joinedload(UserOuiPosition.oui).selectinload(OrgUnitInstance.parents),
        )
        .filter(User.id == payload["sub"])
        .first()
    )
    if not user or user.status != "active":
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user