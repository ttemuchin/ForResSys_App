import os
from pathlib import Path

# dev / prod
def is_dev_mode():
    parent_dir = Path(__file__).parent.parent
    return (parent_dir / "CMakeLists.txt").exists()

BASE_DIR = Path(__file__).parent

if is_dev_mode():
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data" 
    LOGS_DIR = BASE_DIR / "logs"
else:
    APP_DATA_DIR = Path(os.getenv('APPDATA')) / "ResSysApp"
    APP_DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR = APP_DATA_DIR / "models"
    DATA_DIR = APP_DATA_DIR / "data"
    LOGS_DIR = APP_DATA_DIR / "logs"

for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

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