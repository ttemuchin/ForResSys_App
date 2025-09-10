import sys
import os
import signal
sys.path.append(os.path.join(os.path.dirname(__file__), 'site-packages'))
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from datetime import datetime

from config import HOST, PORT, DEBUG, MODEL_CONFIG
from moked_model import model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PY_Server")

# Модели Pydantic для валидации
class PredictionRequest(BaseModel):
    input_data: str = Field(..., description="Входные данные для обработки")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Дополнительные параметры обработки"
    )

class BatchPredictionRequest(BaseModel):
    inputs: List[str] = Field(..., description="Список входных данных")
    batch_size: Optional[int] = Field(default=10, ge=1, le=100)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    server_time: str
    model_info: Dict[str, Any]

# init FastAPI
app = FastAPI(
    title="ML Model Server",
    description="Сервер для обработки ML моделей",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "ML Model Server is running"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка статуса сервера и модели"""
    return {
        "status": "healthy",
        "model_loaded": model.is_loaded,
        "server_time": datetime.now().isoformat(),
        "model_info": model.get_model_info()
    }

@app.post("/predict", summary="Одиночное предсказание")
async def predict(request: PredictionRequest):
    """Обработка одиночного запроса"""
    try:
        logger.info(f"Received prediction request: {request.input_data}")
        
        # Обработка через модель
        result = model.process_batch([request.input_data])[0]
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )

@app.post("/predict/batch", summary="Пакетная обработка")
async def predict_batch(request: BatchPredictionRequest):
    """Обработка батча запросов"""
    try:
        logger.info(f"Received batch request with {len(request.inputs)} items")
        
        if len(request.inputs) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size too large. Max 100 items."
            )
        
        # Обработка батча
        results = model.process_batch(request.inputs)
        
        return {
            "batch_id": f"batch_{datetime.now().timestamp()}",
            "processed_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing error: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """Получение информации о модели"""
    return model.get_model_info()

@app.get("/config")
async def get_server_config():
    """Получение конфигурации сервера"""
    return {
        "server": {
            "host": HOST,
            "port": PORT,
            "debug": DEBUG
        },
        "model_config": MODEL_CONFIG
    }
@app.post("/shutdown")
async def shutdown_server():
    """Эндпоинт для graceful shutdown"""
    os.kill(os.getpid(), signal.SIGINT)
    return {"message": "Server shutting down..."}

if __name__ == "__main__":
    logger.info(f"Starting ML Server on {HOST}:{PORT}")
    logger.info(f"Docs available at: http://{HOST}:{PORT}/docs")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info" if not DEBUG else "debug",
        reload=DEBUG
    )