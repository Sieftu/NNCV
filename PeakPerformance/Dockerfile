FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir pillow transformers timm

WORKDIR /app
COPY model.py /app/model.py
COPY predict.py /app/predict.py
COPY segformer_config/ /app/segformer_config/
COPY model.pt /app/model.pt

ENTRYPOINT ["python", "/app/predict.py"]
