import time
import random
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime

from pydantic import ConfigDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model")

class MockMLModel:
    model_config = ConfigDict(protected_namespaces=())
    
    def __init__(self):
        self.model_name = "MockSignalProcessor"
        self.version = "1.0.0"
        self.is_loaded = True
        logger.info(f"Initialized {self.model_name} v{self.version}")
    
    def preprocess_input(self, input_data: Any) -> Dict:
        """Мокнутый препроцессинг входных данных"""
        return {
            "original": input_data,
            "processed": f"preprocessed_{input_data}",
            "timestamp": datetime.now().isoformat()
        }
    
    def mock_prediction(self, processed_data: Dict) -> Dict:
        """Мокнутое предсказание модели"""
        # Имитация вычислений
        time.sleep(0.1)  # Имитация задержки
        
        return {
            "prediction": random.uniform(0.0, 1.0),
            "confidence": random.uniform(0.7, 0.99),
            "classes": [f"class_{i}" for i in range(5)],
            "probabilities": [random.uniform(0.0, 1.0) for _ in range(5)],
            "processing_time_ms": random.randint(10, 100)
        }
    
    def process_batch(self, inputs: List[Any]) -> List[Dict]:
        """Обработка батча данных"""
        results = []
        for input_data in inputs:
            processed = self.preprocess_input(input_data)
            prediction = self.mock_prediction(processed)
            
            results.append({
                "input": input_data,
                "processed": processed,
                "prediction": prediction,
                "metadata": {
                    "model": self.model_name,
                    "version": self.version,
                    "timestamp": datetime.now().isoformat()
                }
            })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Информация о модели"""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "is_loaded": self.is_loaded,
            "input_size": "variable",
            "output_size": 5,
            "supported_operations": ["predict", "batch_predict", "info"]
        }

model = MockMLModel()