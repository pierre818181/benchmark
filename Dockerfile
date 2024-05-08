FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY . .

RUN apt-get update && apt-get install -y \
    git \
    wget \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/miniconda.sh && \
    bash /workspace/miniconda.sh -b -u -p /workspace && \
    rm -rf /workspace/miniconda.sh

RUN wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

RUN /workspace/bin/conda init bash

RUN /workspace/bin/conda create -n axolotlenv python=3.10 -y && \
    /workspace/bin/conda run -n axolotlenv pip install packaging torch ninja -v && \
    /workspace/bin/conda run -n axolotlenv pip install -e axolotl/[flash-attn,deepspeed] -v && \
    /workspace/bin/conda run -n axolotlenv pip install -r requirements.txt && \
    /workspace/bin/conda run -n axolotlenv pip uninstall flash-attn -v -y && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE /workspace/bin/conda run -n axolotlenv pip install flash-attn -v

RUN /workspace/bin/conda create -n vllmenv python=3.10 -y && \
    /workspace/bin/conda run -n vllmenv pip install -e vllm/ -v && \
    /workspace/bin/conda run -n vllmenv pip install -r requirements.txt

CMD ["python3", "script.py"]