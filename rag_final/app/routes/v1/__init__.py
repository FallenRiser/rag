"""
Aggregates all v1 routers into a single APIRouter mounted at /v1.
"""

from fastapi import APIRouter

from app.routes.v1.endpoints.documents import router as documents_router
from app.routes.v1.endpoints.health import router as health_router
from app.routes.v1.endpoints.ingest import router as ingest_router
from app.routes.v1.endpoints.pipelines import router as pipelines_router
from app.routes.v1.endpoints.query import router as query_router

v1_router = APIRouter(prefix="/v1")

v1_router.include_router(ingest_router)
v1_router.include_router(query_router)
v1_router.include_router(documents_router)
v1_router.include_router(pipelines_router)
v1_router.include_router(health_router)
