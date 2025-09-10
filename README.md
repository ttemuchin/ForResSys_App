RES SYS CONSOLE APP

======/Последовательность работы/======
Запуск приложения:
Пользователь запускает app.exe
EXE файл запускает встроенный Python с FastAPI сервером = Сервер стартует на localhost:8000

Инициализация:

FastAPI сервер загружает ML модель и веса

Сервер регистрирует endpoints для ПРЕДСКАЗАНИЙ

Клиент проверяет доступность сервера через /health

Работа:

Клиент отправляет данные → /predict endpoint

FastAPI вызывает ML модель → получает результат

Результат возвращается клиенту в JSON формате

Завершение:
Клиент отправляет сигнал остановки(ПРИНУДИТЕЛЬНЫЙ) = Сервер сохраняет состояние (если нужно)
= Все процессы корректно завершаются

АРХИТЕКТУРА

C++ Клиент (GUI/Console) 
    ← HTTP/REST API → 
FastAPI Сервер (Python) 
    ← Direct Calls → 
ML Модель (PyTorch)
    ← File I/O → 
Веса моделей (.pth)

ML_Application/
├── 📂 build/ 
├── 📂 client/             # Главная папка client приложения
│   ├── ml_app.cpp         # C++ клиент
│   ├── http_client.cpp    # HTTP клиент
│   ├── http_client.h      # Заголовок HTTP клиента
│   ├── config_loader.cpp  # Загрузчик конфигурации
│   ├── config_loader.h    # Заголовок конфигурации
│   ├── app_config.ini     # Конфигурация
│   └── 📂 resources/      # Ресурсы
│
├── 📂 python_server/       # Автономный Python + FastAPI
│   ├── python.exe          # Portable (embedded) Python интерпретатор
│   ├── 📂 Scripts/         # Исполняемые скрипты
│   ├── 📂 Lib/             # Стандартная библиотека
│   ├── 📂 site-packages/   # Все зависимости
│   │   ├── fastapi/
│   │   ├── uvicorn/
│   │   ├── torch/
│   │   ├── numpy/
│   │   └── ...
│   ├── main.py           # FastAPI сервер
│   ├── 📂 model_storage    # Всё про модели
│   ├── ml_model.py         # ML модель
│   └── requirements.txt    # Список зависимостей
│
├── 📂 models/              # Веса и конфиги моделей
│   ├── model_weights.pth
│   ├── model_config.json
│   └── preprocessing.pkl   # Препроцессинг ???
│
├── 📂 data/                # Примеры данных
│   └── sample_input.txt
│
├── 📂 logs/                # Логи (создается при первом запуске)
│
└── 📄 install.bat          # Скрипт установки 
