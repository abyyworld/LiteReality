FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_DISABLE_XET=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt

# Preserve the CUDA-enabled torch supplied by the base image. All packages are wheels, so this
# image builds entirely on CPU; the Modal function receives a GPU only while serving requests.
RUN python -m pip install --no-cache-dir --no-deps torchvision==0.20.1 && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /root
