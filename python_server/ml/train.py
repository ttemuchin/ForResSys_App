import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os, sys
from pathlib import Path
from sklearn.metrics import r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from Dataset import DynamicNMRDataset
from ConvLayers_model import DynamicNMRRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preproc')))
from Preprocess import parse_data_file, splitSamples, split_data

def parse_config(config_path):
    """Парсинг конфигурационного файла"""
    config = {}
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        return config
    except Exception as e:
        raise Exception(f"Error parsing config file: {str(e)}")

def validate_config(config, parsed_data):
    """Проверка соответствия конфига данным"""
    required_fields = ['name', 'num_samples', 'num_targets_y', 'num_features_x', 'x_lengths']
    for field in required_fields:
        if field not in config:
            raise Exception(f"Missing required field in config: {field}")
    
    # Проверяем количество образцов
    num_samples = int(config['num_samples'])
    if len(parsed_data) != num_samples:
        raise Exception(f"Sample count mismatch: config has {num_samples}, data has {len(parsed_data)}")
    
    # Проверяем количество целевых переменных
    num_targets_y = int(config['num_targets_y'])
    if 'Yi' not in parsed_data[0] or len(parsed_data[0]['Yi']) != num_targets_y:
        raise Exception(f"Target variables count mismatch: config has {num_targets_y}, data has {len(parsed_data[0].get('Yi', []))}")
    
    # Проверяем количество признаков и их длины
    num_features_x = int(config['num_features_x'])
    x_lengths = list(map(int, config['x_lengths'].split(',')))
    
    if len(x_lengths) != num_features_x:
        raise Exception(f"Features count mismatch: config has {num_features_x}, x_lengths has {len(x_lengths)}")
    
    # Проверяем фактические длины признаков в данных
    for i, length in enumerate(x_lengths):
        feature_key = f"X[{i}]"
        if feature_key not in parsed_data[0]:
            raise Exception(f"Feature {feature_key} not found in data")
        if len(parsed_data[0][feature_key]) != length:
            raise Exception(f"Feature {feature_key} length mismatch: config has {length}, data has {len(parsed_data[0][feature_key])}")
    
    # TODO: Добавить проверку y_precision когда будет с чем сравнивать
    if 'y_precision' in config:
        y_precision = list(map(float, config['y_precision'].split(',')))
        if len(y_precision) != num_targets_y:
            raise Exception(f"Y precision count mismatch: config has {len(y_precision)}, targets has {num_targets_y}")
    
    return True

def get_model_path(base_name, model_name, models_dir):
    """Генерируем путь для сохранения весов"""
    filename = f"{base_name}_{model_name}.pth"
    return Path(models_dir) / filename

def get_best_model_path(base_name, model_name, models_dir):
    """Генерируем путь для временного best_model"""
    filename = f"{base_name}_{model_name}_best.pth"
    return Path(models_dir) / filename

def create_model(model_name, input_dims, num_targets):
    """Выбор модели по названию"""
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

def train(base_name, path_to_base, path_to_config, model_name):
    # Создаем папку для моделей в AppData
    models_dir = Path(os.getenv('APPDATA')) / "ResSysApp" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Парсим конфиг
    config = parse_config(path_to_config)
    
    # Парсим данные
    parsed_data = parse_data_file(path_to_base)
    
    # Проверяем соответствие конфига и данных
    validate_config(config, parsed_data)
    
    # Разделяем данные
    train_data, test_data = split_data(parsed_data, train_ratio=0.85, shuffle=True, random_seed=42)
    x_train, y_train = splitSamples(train_data)
    x_test, y_test = splitSamples(test_data)

    # Подготовка данных
    batch_size = 32
    train_dataset = DynamicNMRDataset(*x_train, y=y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = DynamicNMRDataset(*x_test, y=y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dims = [len(x[0]) for x in x_train]
    num_targets = len(y_train[0])

    assert input_dims == [len(x[0]) for x in x_test] and num_targets == len(y_test[0]), "Несоответствие размеров train/test"
    # print(f"Input dimensions: {input_dims}, Number of targets: {num_targets}")

    # Устройство и модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Используемое устройство:", device)

    model = create_model(model_name, input_dims, num_targets)
    model.to(device)

    # Обучение
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best_test_loss = float('inf')
    best_r2 = -float('inf')
    best_epoch = 0
    patience = 10
    counter = 0
    r2_threshold = 0.8

    history = {
        'train_loss': [],
        'test_loss': [],
        'r2': [],
        'best_epoch_data': None
    }

    final_weights_path = get_model_path(base_name, model_name, models_dir)
    best_weights_path = get_best_model_path(base_name, model_name, models_dir)

    for epoch in range(250):
        model.train()
        train_running_loss = 0.0
        
        for batch in train_dataloader:
            *x_batch, y_batch = batch
            x_batch = [x.to(device) for x in x_batch]
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(*x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()
        
        # Валидация
        model.eval()
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

        train_loss = train_running_loss / len(train_dataloader)
        test_loss = test_running_loss / len(test_dataloader)
        r2 = r2_score(all_targets, all_preds)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['r2'].append(r2)
        
        # print(f"Epoch {epoch + 1}")
        # print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | R² Score: {r2:.4f}")
        
        # Early Stopping
        if test_loss < best_test_loss and r2 >= r2_threshold:
            best_test_loss = test_loss
            best_r2 = r2
            best_epoch = epoch
            counter = 0
            
            history['best_epoch_data'] = {
                'predictions': all_preds,
                'targets': all_targets,
                'epoch': epoch
            }

            # Сохраняем лучшие веса во временный файл
            torch.save(model.state_dict(), best_weights_path)
        else:
            counter += 1
            if counter >= patience and r2 >= r2_threshold:
                # print(f"Early stopping at epoch {epoch + 1}")
                # print(f"Best epoch: {best_epoch + 1} | Best Test Loss: {best_test_loss:.4f} | Best R²: {best_r2:.4f}")
                break
        
        # print("-" * 50)

    print(f"Best epoch: {best_epoch + 1} | Best Test Loss: {best_test_loss:.4f} | Best R²: {best_r2:.4f}")

    # Загружаем лучшие веса и сохраняем финальную модель
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, weights_only=True))
        # Удаляем временный файл
        best_weights_path.unlink()

    # Сохраняем финальные веса
    torch.save(model.state_dict(), final_weights_path)
    
    return best_test_loss, str(final_weights_path), best_r2