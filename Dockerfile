FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "if [ \"$SERVICE_ROLE\" = \"monitor\" ]; then exec python -u -m monitor.worker; else exec python -m gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 60 wsgi:app; fi"]
RUN apt-get update && apt-get install -y gcc g++ make && rm -rf /var/lib/apt/lists/*
