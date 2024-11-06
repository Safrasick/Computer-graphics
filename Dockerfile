
FROM python:3.9-slim

# Установка зависимостей для OpenCV, включая libGL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект в контейнер
COPY . /app
WORKDIR /app

# Запуск приложения
CMD ["python", "app.py"]
