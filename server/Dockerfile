# 1 
FROM python:3.8-slim

# 2
RUN pip install Flask gunicorn Pillow torch --no-cache-dir torchvision timm flask_cors tensorflow --no-cache-dir

# 3
COPY src/ /app
WORKDIR /app

# 4
ENV PORT 8080

# 5
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app