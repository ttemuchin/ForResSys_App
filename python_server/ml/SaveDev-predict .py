import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from Dataset import DynamicNMRDataset
from ConvLayers_model import DynamicNMRRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preproc')))
from Preprocess import parse_data_file, splitSamples, split_data

def pred(file_path, model_name, base_name):
    test_data = parse_data_file(file_path) #SygnalsWithoutNoise
    x_test, y_test = splitSamples(test_data)

    batch_size = 32
    test_dataset = DynamicNMRDataset(*x_test, y=y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Используемое устройство:", device)

    # init model
    # model = DynamicNMRRegressor(...)
    model.load_state_dict(torch.load('model_weights_100noise003.pth', weights_only=True)) #в зависимости от обучающей базы
    model.to(device)
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

    # предсказания и цели
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    test_loss = test_running_loss / len(test_dataloader)
