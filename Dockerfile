FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "webui.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]