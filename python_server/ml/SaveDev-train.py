import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import os, sys

from sklearn.metrics import r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from Dataset import DynamicNMRDataset

from ConvLayers_model import DynamicNMRRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preproc')))
from Preprocess import parse_data_file, splitSamples, split_data

def train(base_name, path_to_base, path_to_config, model_name):
    file_path = path_to_base #"./kozlopan_ml100.txt" # ФАЙЛ ОБУЧАЮЩЕЙ ВЫБОРКИ
    parsed_data = parse_data_file(file_path)

    # ПРОВЕРКА
    for i, sample in enumerate(parsed_data[:2]):
        print(f"Sample {i}:")
        print(f"Yi: {sample.get('Yi', [])}")
        for key in sorted(sample.keys()):
            if key.startswith("X["):
                print(f"{key}: first 5 values - {sample[key][:5]}")
        print()
    
    train_data, test_data = split_data(parsed_data, train_ratio=0.85, shuffle=True, random_seed=42)

    x_train, y_train = splitSamples(train_data)
    x_test, y_test = splitSamples(test_data)

    batch_size = 32
    train_dataset = DynamicNMRDataset(*x_train, y=y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = DynamicNMRDataset(*x_test, y=y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dims = [len(x[0]) for x in x_train]  # Длины каждого X[i]
    num_targets = len(y_train[0])  # Количество целевых переменных (Yi)

    assert input_dims == [len(x[0]) for x in x_test] and num_targets == len(y_test[0]), "Несоответствие размеров train/test"
    print(input_dims, num_targets)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Используемое устройство:", device)

    model = DynamicNMRRegressor(input_dims, num_targets)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)



    best_test_loss = float('inf')
    best_r2 = -float('inf')
    best_epoch = 0
    patience = 10  # n эпох без улучшения перед остановкой
    counter = 0
    r2_threshold = 0.8

    history = {
        'train_loss': [],
        'test_loss': [],
        'r2': [],
        'best_epoch_data': None
    }

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
        
        # =========

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
        
        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | R² Score: {r2:.4f}")
        
        # ========= Early Stopping =========
        if test_loss < best_test_loss and r2 >= r2_threshold:  # test_loss главный параметр, R² threshold
            best_test_loss = test_loss
            best_r2 = r2
            best_epoch = epoch
            counter = 0
            
            history['best_epoch_data'] = {
                'predictions': all_preds,
                'targets': all_targets,
                'epoch': epoch
            }

            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience and r2 >= r2_threshold:
                print(f"Early stopping at epoch {epoch + 1}")
                print(f"Best epoch: {best_epoch + 1} | Best Test Loss: {best_test_loss:.4f} | Best R²: {best_r2:.4f}")
        
        print("-" * 50)
    print(f"Best epoch: {best_epoch + 1} | Best Test Loss: {best_test_loss:.4f} | Best R²: {best_r2:.4f}")

    model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    torch.save(model.state_dict(), 'model_weights_100.pth')    

