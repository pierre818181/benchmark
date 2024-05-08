# IMPORTANT

- Please use the pierre/pin-axolotl branch for future use.

# Initial setup

1. Download vllm and axolotl to the root of this project:

```bash
    git clone https://github.com/OpenAccess-AI-Collective/axolotl.git
    git clone https://github.com/vllm-project/vllm.git
```

2. Docker build:
This is the standard docker build. Use Runpod's dockerhub to build and push models.
```
    sudo docker build --no-cache -t benchmark:latest .
```
There are a few different types of docker that you can build. For a heavier version (which optimizes for runtime), you can build using the Dockerfile.heavy file. For H100s, they do not work with the standard mpi4py installation, so we need to build using the Dockerfile.H100 file.

3. How to test shell commands for the Dockerfile and the script?
To get commands for your script, ssh into a docker container with the base image. Run your commands there and then copy those commands over to the script.

4. How to select the right models?
Use this HF spaces to pick the right model for inference and fine-tuning:

```bash
    https://huggingface.co/spaces/hf-accelerate/model-memory-usage
```

## Known issues
- For extremely small machine sizes (<= 2x A2000), the benchmark will not work. This is because installing vllm requires some amount of RAM and these machines might not have it. But we might never want to accept machines smaller than 3xA2000.
- Does not work with V100s because they are quite old and do not support some key functionalities.

## Helpful scripts

```bash
MACHINE_BENCHMARK_AWS_SECRET_ACCESS_KEY=asdad MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID=asdasd MACHINE_BENCHMARK_TOPIC_ARN=asdasd HF_TOKEN=asdad RUN_INFERENCE=false RUNPOD_GPU_COUNT=8 RUNPOD_POD_ID=212313 HF_TOKEN=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 python3 script.py
```

```bash
    sudo docker run -e MACHINE_BENCHMARK_AWS_SECRET_ACCESS_KEY=asdad \
-e MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID=asdasd \
-e MACHINE_BENCHMARK_TOPIC_ARN=asdasd \
-e HF_TOKEN=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 \
-e RUN_INFERENCE=false \
-e RUNPOD_GPU_COUNT=8 \
-e RUNPOD_POD_ID=212313 \
    --gpus all pierre781/benchmark:latest
```

```bash
    conda run -n axolotlenv accelerate launch -m axolotl.cli.train axolotl/examples/openllama-3b/lora.yml
```

Include this in the docker image:
```bash
    conda run -n axolotl pip install flash_attn -U --force-reinstall
```

```bash
    conda run -n axolotl python -m axolotl.cli.preprocess axolotl/examples/openllama-3b/config.yml
```