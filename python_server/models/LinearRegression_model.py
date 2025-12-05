import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicNMR_LinearRegression(nn.Module):
    def __init__(self, input_dims: list, num_targets: int):
        """
        :param input_dims: Список длин сигналов (например, [1000, 2000])
        :param num_targets: Количество целевых переменных (N)
        """
        super().__init__()
        self.num_experiments = len(input_dims)
        self.num_targets = num_targets
        self.input_dims = input_dims
        
        # Общая сумма размеров всех сигналов
        total_input_size = sum(input_dims)
        
        # Простая линейная регрессия
        self.linear_layers = nn.Sequential(
            nn.Linear(total_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_targets)
        )
        
        # Инициализация весов для стабильности
        self._initialize_weights()

    def _initialize_weights(self):
        """Инициализация весов для предотвращения NaN/Inf"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, *x_signals):
        """
        :param x_signals: M сигналов размером [B, L_i], где L_i может отличаться
        :return: Предсказания размером [B, num_targets]
        """
        # Проверка количества входных сигналов
        if len(x_signals) != self.num_experiments:
            raise ValueError(f"Expected {self.num_experiments} signals, got {len(x_signals)}")
        
        # Проверка размеров сигналов
        batch_size = x_signals[0].shape[0]
        
        # Собираем все сигналы в один вектор
        signal_vectors = []
        for i, signal in enumerate(x_signals):
            # Проверяем размерность
            if signal.shape[-1] != self.input_dims[i]:
                # Автоматически обрезаем или дополняем
                if signal.shape[-1] > self.input_dims[i]:
                    signal = signal[..., :self.input_dims[i]]
                else:
                    pad_size = self.input_dims[i] - signal.shape[-1]
                    signal = F.pad(signal, (0, pad_size), mode='constant', value=0)
            
            # Выравниваем сигнал
            flattened = signal.reshape(batch_size, -1)
            signal_vectors.append(flattened)
        
        # Конкатенация всех сигналов
        combined = torch.cat(signal_vectors, dim=1)  # [B, total_input_size]
        
        # Проходим через линейные слои
        return self.linear_layers(combined)

    def get_feature_importance(self, *x_signals):
        """
        Возвращает важность признаков для интерпретации модели
        """
        with torch.no_grad():
            # Получаем веса первого слоя
            first_layer = self.linear_layers[0]
            weights = first_layer.weight.data  # [128, total_input_size]
            
            # Абсолютные значения весов как мера важности
            importance = torch.abs(weights).mean(dim=0)  # [total_input_size]
            
            # Разделяем важность по сигналам
            importance_per_signal = []
            start_idx = 0
            for dim in self.input_dims:
                end_idx = start_idx + dim
                signal_importance = importance[start_idx:end_idx]
                importance_per_signal.append(signal_importance)
                start_idx = end_idx
            
            return importance_per_signal