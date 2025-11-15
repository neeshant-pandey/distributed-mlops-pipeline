"""
FastAPI serving application for DeepLOB model.
"""

import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import PlainTextResponse

from src.serving.predictor import LOBPredictor

# Create FastAPI app
app = FastAPI(
    title="DeepLOB Prediction API",
    description="API for predicting Limit Order Book mid-price movements",
    version="1.0.0",
)

# Prometheus metrics
PREDICTION_COUNT = Counter(
    "prediction_count_total", "Total number of predictions", ["endpoint", "predicted_class"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds", ["endpoint"]
)
MODEL_LOAD_TIME = Gauge("model_load_time_seconds", "Model load time in seconds")
ACTIVE_REQUESTS = Gauge("active_requests", "Number of active requests")

# Global predictor (loaded on startup)
predictor: Optional[LOBPredictor] = None


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Prediction request schema."""

    features: List[List[float]] = Field(
        ...,
        description="LOB features of shape (window_size, n_features)",
        example=[[0.1] * 144] * 100,
    )
    return_probabilities: bool = Field(
        default=False, description="Whether to return class probabilities"
    )


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    predicted_class: int = Field(..., description="Predicted class index (0, 1, 2)")
    predicted_label: str = Field(..., description="Predicted label (down, stationary, up)")
    confidence: float = Field(..., description="Confidence score")
    probabilities: Optional[dict] = Field(
        None, description="Class probabilities if requested"
    )
    prediction_time_ms: float = Field(..., description="Prediction time in milliseconds")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""

    features_batch: List[List[List[float]]] = Field(
        ..., description="Batch of LOB features"
    )
    return_probabilities: bool = Field(default=False)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""

    predictions: List[PredictionResponse]
    total_prediction_time_ms: float


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    timestamp: str
    model_loaded: bool


# API endpoints
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor

    # In production, this should come from environment variables
    model_path = "checkpoints/best_model.pt"

    print(f"Loading model from {model_path}...")
    start_time = time.time()

    try:
        predictor = LOBPredictor(
            model_path=model_path, device="cpu"  # Use CPU for serving
        )
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)
        print(f"✓ Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("⚠ API will start but predictions will fail until model is loaded")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "DeepLOB Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_batch": "/predict/batch (POST)",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if predictor is not None else "model_not_loaded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction.

    Args:
        request: Prediction request with features

    Returns:
        Prediction response with class and confidence
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ACTIVE_REQUESTS.inc()

    try:
        # Validate input shape
        features = np.array(request.features, dtype=np.float32)

        if features.shape != (100, 144):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input shape. Expected (100, 144), got {features.shape}",
            )

        # Predict
        start_time = time.time()
        result = predictor.predict(
            features, return_probabilities=request.return_probabilities
        )
        prediction_time = (time.time() - start_time) * 1000  # Convert to ms

        # Update metrics
        PREDICTION_COUNT.labels(
            endpoint="predict", predicted_class=result["predicted_label"]
        ).inc()
        PREDICTION_LATENCY.labels(endpoint="predict").observe(prediction_time / 1000)

        # Add timing to response
        result["prediction_time_ms"] = prediction_time

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.

    Args:
        request: Batch prediction request

    Returns:
        Batch prediction response
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ACTIVE_REQUESTS.inc()

    try:
        # Convert to numpy
        features_batch = np.array(request.features_batch, dtype=np.float32)

        # Predict
        start_time = time.time()
        results = predictor.predict_batch(
            features_batch, return_probabilities=request.return_probabilities
        )
        total_time = (time.time() - start_time) * 1000

        # Update metrics
        for result in results:
            PREDICTION_COUNT.labels(
                endpoint="predict_batch", predicted_class=result["predicted_label"]
            ).inc()

        PREDICTION_LATENCY.labels(endpoint="predict_batch").observe(total_time / 1000)

        # Add timing to each prediction
        for result in results:
            result["prediction_time_ms"] = total_time / len(results)

        return {
            "predictions": results,
            "total_prediction_time_ms": total_time,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest())


if __name__ == "__main__":
    # Run server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
