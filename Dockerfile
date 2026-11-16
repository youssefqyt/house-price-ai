FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY main.py ./main.py
COPY data ./data
RUN useradd --create-home --uid 10001 appuser && mkdir -p models && chown appuser:appuser models
USER appuser
ENTRYPOINT ["python", "main.py"]
CMD ["evaluate"]
