# NVIDIA NGC TensorFlow image (Feb 2025 build, TF2, Python3, GPU support)
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set working directory
WORKDIR /app

# Environment variables
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (including Jupyter)
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Configure Jupyter Notebook
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default command: start Jupyter
CMD ["jupyter", "notebook", "--notebook-dir=/app", "--ip=0.0.0.0", "--allow-root", "--no-browser"]