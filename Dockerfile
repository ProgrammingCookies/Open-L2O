FROM python:3.12-slim

# build-essential: compile any pip package without a prebuilt Linux/py3.12 wheel
# graphviz: system `dot` binary needed by torchviz/graphviz (Model_Free_L2O/L2O-Minimax)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# TensorFlow (Model_Base_L2O, all Model_Free_L2O libs except L2O-Minimax, Benchmarking)
# and PyTorch (Model_Free_L2O/L2O-Minimax) each pin their own conflicting nvidia-*-cu12
# CUDA/cuDNN pip packages, so they get separate virtualenvs instead of fighting over
# the same site-packages (which silently breaks TF's GPU detection).
COPY requirements-tensorflow.txt requirements-pytorch.txt ./

RUN python -m venv /opt/venv-tf \
    && /opt/venv-tf/bin/pip install --no-cache-dir -r requirements-tensorflow.txt

RUN python -m venv /opt/venv-torch \
    && /opt/venv-torch/bin/pip install --no-cache-dir -r requirements-pytorch.txt

# pip's CUDA/cuDNN wheels aren't found via RPATH alone -- each venv's nvidia-*-cu12
# lib dirs must be on LD_LIBRARY_PATH for the framework to dlopen them successfully.
ENV LD_LIBRARY_PATH="/opt/venv-tf/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv-tf/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/opt/venv-tf/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/venv-tf/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/venv-tf/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv-tf/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/venv-tf/lib/python3.12/site-packages/nvidia/curand/lib:/opt/venv-tf/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/venv-tf/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/venv-tf/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/venv-tf/lib/python3.12/site-packages/nvidia/nvjitlink/lib"

# Wrapper for L2O-Minimax (PyTorch): points LD_LIBRARY_PATH at venv-torch's own
# CUDA/cuDNN libs instead of venv-tf's, then execs venv-torch's python.
RUN printf '%s\n' \
    '#!/bin/bash' \
    'export LD_LIBRARY_PATH="/opt/venv-torch/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv-torch/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/opt/venv-torch/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/venv-torch/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/venv-torch/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv-torch/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/venv-torch/lib/python3.12/site-packages/nvidia/curand/lib:/opt/venv-torch/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/venv-torch/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/venv-torch/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/venv-torch/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/opt/venv-torch/lib/python3.12/site-packages/nvidia/nvtx/lib"' \
    'exec /opt/venv-torch/bin/python "$@"' \
    > /usr/local/bin/torch-python \
    && chmod +x /usr/local/bin/torch-python

# Default shell uses the TensorFlow env (everything except L2O-Minimax).
# For L2O-Minimax, run: torch-python your_script.py
ENV PATH="/opt/venv-tf/bin:$PATH"

CMD ["bash"]
