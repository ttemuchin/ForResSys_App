from pathlib import Path
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
import ml.predict as PRED
import ml.train as TRAIN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PY_Server")

# Модели Pydantic для валидации
# ???
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

@app.get("/model/info")
async def get_model_info():
    """Получение информации о модели"""
    return model.get_model_info()

@app.post("/train")
async def train_model(request: dict):
    try:
        base_name = request.get("base_name")
        base_path = request.get("base_path") 
        config_path = request.get("config_path")
        model_type = request.get("model_type")
        
        best_loss, weights_path, accuracy = TRAIN.train(base_name, base_path, config_path, model_type)
        
        return {
            "status": "success",
            "message": f"Training completed for {base_name}",
            "model_type": model_type,
            "accuracy": accuracy,
            "best_loss": best_loss,
            "weights_path": weights_path
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict")
async def predict_with_model(request: dict):
    try:
        file_path = request.get("file_path")
        model_name = request.get("model_name") 
        base_name = request.get("base_name")
        
        if not file_path or not model_name or not base_name:
            return {"status": "error", "message": "Missing required parameters"}
        
        output_path, metrics = PRED.pred(file_path, model_name, base_name)
        
        return {
            "status": "success",
            "output_path": output_path,
            "metrics": metrics,
            "message": f"Prediction completed using {model_name} model trained on {base_name}"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
        access_log=False
        # reload=DEBUG 
    )