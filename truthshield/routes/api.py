"""Combined API router assembled from focused route modules."""
from fastapi import APIRouter
from .claims import router as claims_router
from .media import router as media_router
from .urls import router as urls_router
from .telemetry import router as telemetry_router

router = APIRouter()
router.include_router(claims_router)
router.include_router(media_router)
router.include_router(urls_router)
router.include_router(telemetry_router)
