import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Настройки сервера
HOST = "localhost"
PORT = 8000
DEBUG = False

# Настройки модели
MODEL_CONFIG = {
    "input_size": 100,
    "output_size": 10,
    "model_version": "1.0.0",
    "supported_formats": [".txt", ".csv", ".json"]
}

# Создаем необходимые директории
for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)