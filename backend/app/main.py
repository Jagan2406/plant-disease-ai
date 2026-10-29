from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.predict import router as prediction_router
from app.services.swin_sam2 import SwinSAM2Predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        application.state.swin_sam2_predictor = SwinSAM2Predictor.load()
    except Exception:
        logger.exception("Failed to initialize the Swin-SAM2 prediction service.")
        raise
    yield
    application.state.swin_sam2_predictor = None


app = FastAPI(
    title="Plant Disease AI API",
    version="1.0.0",
    description="Plant disease detection, explainability, and treatment recommendation API",
    lifespan=lifespan,
)

app.include_router(prediction_router)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Plant Disease AI API",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }