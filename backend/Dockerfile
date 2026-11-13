FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# INSTALL_DL=true adds TensorFlow for in-app Deep Learning training.
ARG INSTALL_DL=false
COPY requirements.txt requirements-dl.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_DL" = "true" ]; then pip install -r requirements-dl.txt; fi

COPY . .

RUN mkdir -p storage/datasets storage/models storage/vectorstores storage/artifacts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
