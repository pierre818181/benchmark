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

3. Testing
To check if a build works, ssh into a docker container with the base image. Run your commands there and then copy those commands over to a dockerfile.

4. How to select the right models?
Use this HF spaces to pick the right model for inference and fine-tuning:

```bash
    https://huggingface.co/spaces/hf-accelerate/model-memory-usage
```

## Known issues
For extremely small machine sizes (<= 2x A2000), the benchmark will not work. This is because installing vllm requires some amount of RAM and these machines might not have it. But we might never want to accept machines smaller than 3xA2000.
