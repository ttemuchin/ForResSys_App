import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys, os
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from Dataset import DynamicNMRDataset
from ConvLayers_model import DynamicNMRRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preproc')))
from Preprocess import parse_data_file, splitSamples

def get_weights_path(base_name, model_name):
    """Получаем путь к весам модели по названию базы и модели"""
    models_dir = Path(os.getenv('APPDATA')) / "ResSysApp" / "models"
    weights_path = models_dir / f"{base_name}_{model_name}_best.pth"
    
    if not weights_path.exists():
        raise Exception(f"Weights file not found: {weights_path}")
    
    return weights_path

def get_model_config_path(base_name):
    """Получаем путь к конфигу базы"""
    learning_base_dir = Path(os.getenv('APPDATA')) / "ResSysApp" / "data" / "LearningBase"
    config_path = learning_base_dir / "Configs" / f"{base_name}.txt"
    
    if not config_path.exists():
        raise Exception(f"Config file not found: {config_path}")
    
    return config_path

def parse_config(config_path):
    """Парсим конфигурационный файл"""
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config

def create_model(model_name, input_dims, num_targets):
    """Создаем модель по названию"""
    if model_name == "svr":
        # TODO: Заменить на вашу SVR модель
        return DynamicNMRRegressor(input_dims, num_targets)
    elif model_name == "convolutional":
        return DynamicNMRRegressor(input_dims, num_targets)
    elif model_name == "linear_regression":
        # TODO: Заменить на вашу Linear Regression модель
        return DynamicNMRRegressor(input_dims, num_targets)
    else:
        raise Exception(f"Unknown model type: {model_name}")

def save_predictions(all_preds, all_targets, file_path, model_name, base_name, metrics):
    """Сохраняем предсказания в файл"""
    output_dir = Path(os.getenv('APPDATA')) / "ResSysApp" / "data" / "Output"
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
        f.write("Sample\t" + "\t".join([f"Target_{i}" for i in range(len(all_targets[0]))]) + 
                "\t" + "\t".join([f"Pred_{i}" for i in range(len(all_preds[0])+1)]) + "\n")
        
        for i in range(len(all_preds)):
            f.write(f"{i+1}\t")
            f.write("\t".join([f"{val:.6f}" for val in all_targets[i]]) + "\t")
            f.write("\t".join([f"{val:.6f}" for val in all_preds[i]]) + "\n")
    
    return str(output_path)

def pred(file_path, model_name, base_name):
    """Основная функция предсказания"""
    weights_path = get_weights_path(base_name, model_name)
    config_path = get_model_config_path(base_name)
    
    config = parse_config(config_path)
    num_features_x = int(config['num_features_x'])
    x_lengths = list(map(int, config['x_lengths'].split(',')))
    num_targets_y = int(config['num_targets_y'])
    
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
    r2 = r2_score(all_targets, all_preds)
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'test_loss': test_loss
    }
    
    output_path = save_predictions(all_preds, all_targets, file_path, model_name, base_name, metrics)
    
    return output_path, metrics