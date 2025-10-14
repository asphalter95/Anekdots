FROM python:3.12

# Указываем рабочую директорию внутри контейнера
WORKDIR /anekdots

# Сначала копируем requirements.txt (чтобы кешировался pip install)
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install -r requirements.txt

# Копируем всё приложение
COPY . .

# Запускаем приложение
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
