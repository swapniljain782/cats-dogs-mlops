"""FastAPI application for Cats vs Dogs classification."""
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.utils.config import get_config
from src.utils.logging import get_logger, setup_logging, RequestLogger
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ErrorResponse
)
from src.api.model_loader import (
    load_model,
    get_model_version,
    get_class_names,
    predict,
    predict_from_base64
)
from src.monitoring.metrics import (
    record_http_request,
    record_prediction,
    set_model_loaded,
    get_metrics,
    CONTENT_TYPE_LATEST,
)

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Cats vs Dogs API...")
    try:
        load_model()
        set_model_loaded(True, get_model_version())
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        set_model_loaded(False)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Cats vs Dogs API...")


app = FastAPI(
    title="Cats vs Dogs Classification API",
    description="Binary image classification for pet adoption platform",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files directory for the web UI
STATIC_DIR = Path(__file__).parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log requests and responses with timing."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    with RequestLogger(logger, request_id) as req_logger:
        req_logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else "unknown"
        )
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            record_http_request(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
                duration=duration
            )
            
            req_logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2)
            )
            
            return response
        except Exception as e:
            duration = time.time() - start_time
            req_logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=round(duration * 1000, 2)
            )
            raise


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    model_loaded = get_model_version() != "unknown"
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_version=get_model_version(),
        model_loaded=model_loaded
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None, description="Image file"),
    image_base64: Optional[str] = Form(None, description="Base64 encoded image"),
):
    """Predict cat or dog from uploaded image."""
    start_time = time.time()
    
    # Validate input
    if file is None and image_base64 is None:
        raise HTTPException(
            status_code=400,
            detail="Either 'file' or 'image_base64' must be provided"
        )
    
    try:
        if file is not None:
            # Read uploaded file
            image_bytes = await file.read()
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
        else:
            # Decode base64
            image_bytes = image_base64.encode() if isinstance(image_base64, str) else image_base64
        
        # Run prediction
        pred_class, pred_prob, class_probs = predict(image_bytes)
        
        # Record metrics
        latency = time.time() - start_time
        record_prediction(pred_class, latency, pred_prob)
        
        return PredictionResponse(
            class_name=pred_class,
            probability=pred_prob,
            class_probabilities=class_probs,
            model_version=get_model_version()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/base64", response_model=PredictionResponse, tags=["Prediction"])
async def predict_base64_endpoint(request: PredictionRequest):
    """Predict cat or dog from base64 encoded image (JSON)."""
    start_time = time.time()
    
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")
    
    try:
        pred_class, pred_prob, class_probs = predict_from_base64(request.image_base64)
        
        latency = time.time() - start_time
        record_prediction(pred_class, latency, pred_prob)
        
        return PredictionResponse(
            class_name=pred_class,
            probability=pred_prob,
            class_probabilities=class_probs,
            model_version=get_model_version()
        )
    except Exception as e:
        logger.error(f"Base64 prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=get_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Serve the web UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    # Fallback to JSON info if UI not available
    return {
        "name": "Cats vs Dogs Classification API",
        "version": "1.0.0",
        "description": "Binary image classification for pet adoption platform",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_base64": "/predict/base64",
            "metrics": "/metrics",
            "docs": "/docs",
            "ui": "/"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)