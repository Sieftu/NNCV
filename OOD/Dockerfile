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

# Pick which OOD scoring method ships in the image by setting OOD_CONFIG at
# build time. Defaults to msp.
#
#   docker build -t nncv:ood-msp     --build-arg OOD_CONFIG=ood_config_msp.json     -f OOD/Dockerfile OOD/
#   docker build -t nncv:ood-tempmsp --build-arg OOD_CONFIG=ood_config_temp_msp.json -f OOD/Dockerfile OOD/
#   docker build -t nncv:ood-energy  --build-arg OOD_CONFIG=ood_config_energy.json   -f OOD/Dockerfile OOD/
#   docker build -t nncv:ood-pixel   --build-arg OOD_CONFIG=ood_config_pixel_uncertainty.json -f OOD/Dockerfile OOD/
ARG OOD_CONFIG=ood_config_msp.json

WORKDIR /app
COPY model.py /app/model.py
COPY ood_scoring.py /app/ood_scoring.py
COPY predict.py /app/predict.py
COPY segformer_config/ /app/segformer_config/
COPY model.pt /app/model.pt
COPY configs/${OOD_CONFIG} /app/ood_config.json

ENTRYPOINT ["python", "/app/predict.py"]
