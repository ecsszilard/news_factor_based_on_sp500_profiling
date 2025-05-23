FROM nvcr.io/nvidia/tensorflow:24.08-tf2-py3

# Környezeti változók beállítása
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV CUDA_PATH=/usr/local/cuda

# Rendszer függőségek telepítése
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    python3-dev \
	build-essential \
    vim \
    htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python csomagok telepítése
RUN pip install --no-cache-dir --upgrade \
    ipykernel \
    jupyterlab \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-fuzzy \
    nltk \
    nvidia-ml-py

# Alkalmazás mappája
WORKDIR /app

# Alapértelmezett parancs
CMD ["bash"]