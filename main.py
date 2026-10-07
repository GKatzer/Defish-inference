# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import logging

from core.config import settings
from core.globals import models

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context for managing resources"""
    # STARTUP - load the models
    logger.info("Starting ML Service...")
    try:
        # Import here so that the errors are visible
        from core.model_loader import load_detector
        logger.info("Loading ONNX detector model...")
        models["detector"] = load_detector()
        logger.info("Detector loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        import traceback
        traceback.print_exc()
        # Do not crash, but log the error
        models["detector"] = None

    try:
        from core.model_loader import load_classifier
        logger.info("Load ONNX classifier model...")
        models["classifier"] = load_classifier(
            tta=settings.classifier_tta,
            chunk_size=settings.classifier_chunk_size,
        )
        logger.info("Classifier loaded successefully")

    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        import traceback
        traceback.print_exc()
        models["classifier"] = None
    
    logger.info("✅ ML Service started successfully")
    
    yield
    
    # SHUTDOWN - release the resources
    logger.info("Shutting down ML Service...")
    models.clear()

app = FastAPI(
    title="ML Inference Service",
    description="Microservice for an ensemble of neural networks",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production specify concrete domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router, handling import errors
try:
    from api.endpoints import router as api_router
    app.include_router(api_router)
    logger.info("API router loaded")
except Exception as e:
    logger.error(f"Failed to load API router: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for stability
        workers=2,  # ← INCREASED! Was 1, now 2 (by the number of cores)
        worker_class="uvicorn.workers.UvicornWorker",
        timeout_keep_alive=180  # Increased timeout
    )
