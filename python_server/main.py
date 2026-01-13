from pathlib import Path
import sys
import os
import signal
import json
import shutil
from typing import Dict, Any, Optional, List
sys.path.append(os.path.join(os.path.dirname(__file__), 'site-packages'))
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
import math
import uvicorn
import logging
from datetime import datetime

from config import HOST, PORT, DEBUG, MODEL_CONFIG, DATA_DIR, MODELS_DIR, OUTPUT_DIR
import ml.predict as PRED
import ml.train as TRAIN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PY_Server")

# Модели Pydantic для валидации JSON запросов
class BaseConfig(BaseModel):
    name: str
    N: int  # num_samples
    nY: int  # num_targets_y
    accuracy: list[float]
    nX: int  # num_features_x
    dimension: list[int]

class TrainRequest(BaseModel):
    method: str = "train"
    model: str
    baseConfig: BaseConfig
    basePath: str
    hyperparameters: Optional[Dict[str, Any]] = None
    target_metrics: Optional[Dict[str, float]] = None

class PredictRequest(BaseModel):
    method: str = "predict"
    model: str
    baseName: str
    predPath: str
    # output_format: str = "txt"
    # include_metrics: bool = True

class ProcessJsonRequest(BaseModel):
    json_data: Dict[str, Any]

# class HealthResponse(BaseModel):
#     status: str
#     model_loaded: bool
#     server_time: str
#     model_info: Dict[str, Any]

# init FastAPI
app = FastAPI(
    title="ML Model Server",
    description="Сервер для обучения и инференса моделей регрессионного анализа",
    version="2.2.0",
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

@app.get("/health") #, response_model=HealthResponse
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

#  конфиги баз можно сохранять в json
def save_base_config(base_config: BaseConfig, config_dir: Path):
    """Сохранение конфигурации базы в файл"""
    config_path = config_dir / f"{base_config.name}.txt"
    with open(config_path, 'w') as f:
        f.write(f"name={base_config.name}\n")
        f.write(f"num_samples={base_config.N}\n")
        f.write(f"num_targets_y={base_config.nY}\n")
        f.write(f"y_precision={','.join(map(str, base_config.accuracy))}\n")
        f.write(f"num_features_x={base_config.nX}\n")
        f.write(f"x_lengths={','.join(map(str, base_config.dimension))}\n")
    return str(config_path)

def upload_learning_base(base_config: BaseConfig, base_path: str):
    try:
        learning_base_dir = DATA_DIR / "LearningBase"
        config_dir = learning_base_dir / "Configs"
        learning_base_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Training file not found: {base_path}")
        
        dest_path = learning_base_dir / f"{base_config.name}.txt"
        shutil.copy2(base_path, dest_path)
        
        config_path = save_base_config(base_config, config_dir)
        
        logger.info(f"Learning base uploaded: {base_config.name}")
        return {
            "status": "success",
            "base_name": base_config.name,
            "base_path": str(dest_path),
            "config_path": config_path
        }
        
    except Exception as e:
        logger.error(f"Error uploading learning base: {str(e)}")
        return {"status": "error", "message": str(e)}

def internal_train(base_name: str, base_path: str, config_path: str, model_type: str):
    try:
        best_loss, weights_path, best_r2 = TRAIN.train(
            base_name, base_path, config_path, model_type
        )
        
        logger.info(f"\nTrain finished \nBest Loss = {best_loss}\nBest R2-score = {best_r2}\n")
        
        return {
            "status": "success",
            "message": f"Training completed for {base_name}",
            "model_type": model_type,
            "best_r2": float(best_r2) if math.isfinite(best_r2) else 0.0,
            "best_loss": float(best_loss) if math.isfinite(best_loss) else 0.0,
            "weights_path": weights_path
        }
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return {"status": "error", "message": str(e)}

def internal_predict(file_path: str, model_name: str, base_name: str):
    try:
        output_path, metrics = PRED.pred(file_path, model_name, base_name)
        logger.info(f"\nPredict finished\n Metrics:\n{metrics}\n")
        
        return {
            "status": "success",
            "output_path": output_path,
            "metrics": metrics,
            "message": f"Prediction completed using {model_name} model trained on {base_name}"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/process_json")
async def process_json(request: ProcessJsonRequest):
    """Принимает JSON, выполняет train или predict"""
    try:
        json_data = request.json_data
        method = json_data.get("method")
        
        if method == "train":
            # Валидация
            train_req = TrainRequest(**json_data)
            
            # 1
            upload_result = upload_learning_base(
                train_req.baseConfig, 
                train_req.basePath
            )
            
            if upload_result["status"] == "error":
                return upload_result
            
            # 2
            base_name = train_req.baseConfig.name
            base_path = upload_result["base_path"]
            config_path = upload_result["config_path"]
            
            train_result = internal_train(
                base_name, 
                base_path, 
                config_path, 
                train_req.model
            )
            
            return {
                "operation": "train",
                "upload_result": upload_result,
                "train_result": train_result
            }
            
        elif method == "predict":
            # Валидация
            predict_req = PredictRequest(**json_data)
            
            predict_result = internal_predict(
                predict_req.predPath,
                predict_req.model,
                predict_req.baseName
            )
            
            return {
                "operation": "predict",
                "predict_result": predict_result
            }
            
        else:
            return {
                "status": "error",
                "message": f"Unknown method: {method}. Use 'train' or 'predict'"
            }
            
    except Exception as e:
        logger.error(f"Error processing JSON: {str(e)}")
        return {
            "status": "error",
            "message": f"JSON processing error: {str(e)}"
        }

@app.get("/config")
async def get_server_config():
    """Получение конфигурации сервера"""
    return {
        "server": {
            "host": HOST,
            "port": PORT,
            "debug": DEBUG
        },
        "model_config": MODEL_CONFIG,
        "paths": {
            "data_dir": str(DATA_DIR),
            "models_dir": str(MODELS_DIR),
            "weights_dir": str(OUTPUT_DIR)
        }
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