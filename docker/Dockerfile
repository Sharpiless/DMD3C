# Base image with CUDA and PyTorch support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# System setup
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev git build-essential cmake \
    libopenblas-dev libomp-dev g++ curl vim wget libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

RUN conda --version

# Clone DMD3C project
RUN	cd /home \
    && git clone https://github.com/fetty31/DMD3C.git

# Set python alias
RUN ln -s /usr/bin/python3 /usr/bin/python
