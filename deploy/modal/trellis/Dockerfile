FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg git git-lfs libgl1 libglib2.0-0 libjpeg-dev python3.10 \
    python3.10-dev python3-pip python-is-python3 zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir \
      torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

RUN git clone --branch main --recursive https://github.com/microsoft/TRELLIS.2.git /root/TRELLIS.2

# Install pure Python and prebuilt-wheel dependencies on a CPU builder. app.py adds the native
# CUDA extensions as one separately cached GPU-backed image layer.
RUN python3.10 -m pip install --no-cache-dir \
      imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh \
      transformers gradio==6.0.1 tensorboard pandas lpips zstandard \
      git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 \
      pillow-simd kornia timm xformers==0.0.29.post3 huggingface-hub hf-xet

WORKDIR /root
