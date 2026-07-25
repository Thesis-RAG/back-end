"""
Global exception handlers. Attaches structured JSON error responses with trace IDs
to all FastAPI validation errors and unhandled exceptions.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# Register application-wide exception handlers onto the FastAPI instance.
def register_exception_handlers(app: FastAPI):
    # Return a 422 with field-level validation details and the current trace ID.
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.errors(),
                "trace_id": getattr(request.state, "trace_id", None),
            },
        )

    # Return a 500 with a generic message and the current trace ID. Logs the
    # full traceback server-side — without this, unhandled exceptions left
    # zero trace in the logs, only the generic response reached the client.
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        trace_id = getattr(request.state, "trace_id", None)
        logger.exception("Unhandled exception trace_id=%s path=%s", trace_id, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "trace_id": trace_id,
            },
        )
