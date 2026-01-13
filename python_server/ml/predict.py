import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import math
import json
import sys, os
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from Dataset import DynamicNMRDataset
from ConvLayers_model import DynamicNMR_ConvRegressor
from LinearRegression_model import DynamicNMR_LinearRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preproc')))
from Preprocess import parse_data_file, splitSamples

def get_weights_path(base_name, model_name):
    """Путь к весам модели по названию базы и модели"""
    models_dir = Path(os.path.dirname(__file__)).parent.parent / "models"
    weights_path = models_dir / f"{base_name}_{model_name}.pth"
    
    if not weights_path.exists():
        raise Exception(f"Weights file not found: {weights_path}")
    
    return weights_path

def get_model_config_path(base_name):
    """Путь к конфигу базы"""
    learning_base_dir = Path(os.path.dirname(__file__)).parent.parent / "data" / "LearningBase"
    config_path = learning_base_dir / "Configs" / f"{base_name}.json"
    
    if not config_path.exists():
        raise Exception(f"Config file not found: {config_path}")
    
    return config_path

def parse_config(config_path):
    """Парсим JSON конфигурационный файл"""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data

def create_model(model_name, input_dims, num_targets):
    """Создаем модель по названию"""
    if model_name == "svr":
        # TODO: Заменить на SVR модель
        return DynamicNMR_ConvRegressor(input_dims, num_targets)
    elif model_name == "convolutional":
        return DynamicNMR_ConvRegressor(input_dims, num_targets)
    elif model_name == "linear_regression":
        return DynamicNMR_LinearRegression(input_dims, num_targets)
    else:
        raise Exception(f"Unknown model type: {model_name}")

def save_predictions(all_preds, all_targets, file_path, model_name, base_name, metrics):
    """Сохраняем предсказания в файл"""
    # output_dir = Path(os.getenv('APPDATA')) / "ResSysApp" / "data" / "Output"
    output_dir = Path(os.path.dirname(__file__)).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_filename = Path(file_path).stem
    output_filename = f"{input_filename}_{model_name}_{base_name}_out.txt"
    output_path = output_dir / output_filename
    
    with open(output_path, 'w') as f:
        f.write("PREDICTION RESULTS\n")
        f.write("==================\n")
        f.write(f"Input file: {file_path}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Training base: {base_name}\n")
        f.write(f"MSE: {metrics['mse']:.6f}\n")
        f.write(f"Loss: {metrics['test_loss']:.6f}\n")
        f.write(f"R2 Score: {metrics['r2']:.6f}\n")
        f.write("\nPREDICTIONS:\n")
        
        col_width = 12
        
        # Заголовки
        headers = ["Sample"]
        headers.extend([f"Target_{i}" for i in range(len(all_targets[0]))])
        headers.extend([f"Pred_{i}" for i in range(len(all_preds[0]))])
        
        header_line = "".join([h.ljust(col_width) for h in headers])
        f.write(header_line + "\n")
        
        # Данные
        for i in range(len(all_preds)):
            row = [str(i+1)]
            row.extend([f"{val:.6f}" for val in all_targets[i]])
            row.extend([f"{val:.6f}" for val in all_preds[i]])
            
            formatted_row = []
            for item in row:
                formatted_row.append(item.ljust(col_width))

            f.write("".join(formatted_row) + "\n")
    
    return str(output_path)

def safe_r2_score(y_true, y_pred):
    """Безопасное вычисление R2-score"""
    try:
        if (np.any(~np.isfinite(y_true)) or 
            np.any(~np.isfinite(y_pred)) or 
            len(y_true) < 2 or 
            np.var(y_true) == 0):
            return 0.0
        
        r2 = r2_score(y_true, y_pred)
        return r2 if np.isfinite(r2) else 0.0
        
    except:
        return 0.0

def pred(file_path, model_name, base_name):
    """Основная функция предсказания"""
    weights_path = get_weights_path(base_name, model_name)
    config_path = get_model_config_path(base_name)
    
    config = parse_config(config_path)
    num_features_x = config['nX']   #int(config['num_features_x'])
    x_lengths = config['dimension'] #list(map(int, config['x_lengths'].split(',')))
    num_targets_y = config['nY']    #int(config['num_targets_y'])
    
    test_data = parse_data_file(file_path)
    x_test, y_test = splitSamples(test_data)
    
    # Проверки соответствие размеров
    if len(x_test) != num_features_x:
        raise Exception(f"Feature count mismatch: config has {num_features_x}, data has {len(x_test)}")
    
    for i, length in enumerate(x_lengths):
        if len(x_test[i][0]) != length:
            raise Exception(f"Feature X[{i}] length mismatch: config has {length}, data has {len(x_test[i][0])}")
    
    if len(y_test[0]) != num_targets_y:
        raise Exception(f"Target count mismatch: config has {num_targets_y}, data has {len(y_test[0])}")
    
    batch_size = 32
    test_dataset = DynamicNMRDataset(*x_test, y=y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_name, x_lengths, num_targets_y)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    test_running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_dataloader:
            *x_batch, y_batch = batch
            x_batch = [x.to(device) for x in x_batch]
            y_batch = y_batch.to(device)
            
            outputs = model(*x_batch)
            loss = criterion(outputs, y_batch)
            test_running_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    test_loss = test_running_loss / len(test_dataloader)
    
    # метрики
    mse = mean_squared_error(all_targets, all_preds)
    r2 = safe_r2_score(all_targets, all_preds)
    
    metrics = {
        'mse': float(mse) if math.isfinite(mse) else 0.0,
        'r2': float(r2) if math.isfinite(r2) else 0.0,
        'test_loss': test_loss
    }
    
    output_path = save_predictions(all_preds, all_targets, file_path, model_name, base_name, metrics)
    
    return output_path, metrics