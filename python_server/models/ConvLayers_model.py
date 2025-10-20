import torch.nn as nn
import torch

class DynamicNMRRegressor(nn.Module):
    def __init__(self, input_dims: list, num_targets: int, conv_filters: int = 32):
        """
        :param input_dims: Список длин сигналов (например, [1000, 2000])
        :param num_targets: Количество целевых переменных (N)
        :param conv_filters: Базовое число фильтров в сверточных слоях
        """
        super().__init__()
        self.num_experiments = len(input_dims)
        self.num_targets = num_targets
        
        # 1. Подготовка к объединению сигналов
        self.max_len = max(input_dims)
        
        # 2. Общие сверточные слои
        self.shared_conv = nn.Sequential(
            nn.Conv1d(self.num_experiments, conv_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_filters, conv_filters * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        # 3. Вычисление размера выхода сверточной части
        dummy_input = torch.zeros(1, self.num_experiments, self.max_len)
        conv_out_size = self.shared_conv(dummy_input).shape[1]
        
        # 4. Финальные слои
        self.final_fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_targets)
        )

    def forward(self, *x_signals):
        padded_signals = []
        for x in x_signals:
            pad_size = self.max_len - x.shape[-1]
            padded = torch.nn.functional.pad(x, (0, pad_size), mode='constant', value=0)
            padded_signals.append(padded.unsqueeze(1))  # [B, 1, max_len]
        
        combined = torch.cat(padded_signals, dim=1)  # [B, M, max_len]
        
        features = self.shared_conv(combined)
        
        return self.final_fc(features)
