FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

CMD ["python3", "script.py"]