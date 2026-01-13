import os
from pathlib import Path

def get_app_base_dir():
    """Получаем базовую директорию приложения"""
    # Если файл конфига существует рядом, используем его расположение
    config_path = Path(__file__).parent / "app_config.ini"
    if config_path.exists():
        return Path(__file__).parent.parent
    
    # Иначе используем директорию скрипта
    return Path(__file__).parent

BASE_DIR = get_app_base_dir()
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"

for d in [MODELS_DIR, DATA_DIR, LOGS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Настройки сервера
HOST = "localhost"
PORT = 8000
DEBUG = False

# Настройки модели
MODEL_CONFIG = {
    "model_version": "2.2",
    "supported_formats": [".txt"] #".csv", ".json"
}