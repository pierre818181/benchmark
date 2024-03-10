import os
from loguru import logger
import subprocess
from collections import defaultdict
import re
import pynvml
import math

def shell_cmd(command, env={}):
    try:
        all_envs = os.environ.copy()
        if env:
            for k, v in env.items():
                all_envs[k] = v
        logger.info(f"running command: { ' '.join(command) }")
        result = subprocess.run(command, capture_output=True, text=True, env=all_envs)

        if result.returncode != 0:
            logger.error(result.stderr.strip())
            logger.error(f"could not execute shell command: { ' '.join(command) }")
            return None, result.stderr.strip()

        logger.info("Command output:")
        # logger.info(result.stdout.strip())
        # logger.error(result.stderr.strip())
        return result.stdout.strip(), None
    except Exception as e:
        return None, str(e)

def get_parallel_size_accounting_for_attn_head(device_count):
    possible_parallel_size = [32,16,8,4,2,1]
    for p_size in possible_parallel_size:
        if device_count > p_size:
            return p_size
    
    return 1

def run_multi_gpu_inference(device_count):
    command_env = defaultdict(str)
    command_env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    command_env["TOKENIZERS_PARALLELISM"] = "false"
    device_count_attn_heads = get_parallel_size_accounting_for_attn_head(device_count)
    if device_count > 32:
        return None, f"Inference for devices with more than 32 GPUs not supported at once. Current GPU count: { device_count }"
    command = ["python3", "vllm/benchmarks/benchmark_throughput.py", 
            "--dataset=ShareGPT_V3_unfiltered_cleaned_split.json", "--model=mistralai/Mistral-7B-Instruct-v0.1",
            "--tokenizer=hf-internal-testing/llama-tokenizer", "--num-prompts=30", "--dtype=bfloat16", 
            "--max-model-len=8192", "--device=cuda", f"--tensor-parallel-size={device_count_attn_heads}"]
    out, err = shell_cmd(command, env=command_env)
    if err != None:
        return None, err
    else:
        logger.info(out)
        split_sentences = out.split("\n")
        logger.info(split_sentences)
        for sentence in split_sentences:
            # Regex pattern to extract numbers
            pattern = r"Throughput: ([\d.]+) requests/s, ([\d.]+) tokens/s"
            # Search for matches
            match = re.search(pattern, sentence)
            logger.info(match)
            if match:
                logger.info(match)
                logger.info(f"matched sentence: { sentence }")
                throughput = float(match.group(1))
                requests_per_s = float(match.group(2))
                logger.info(throughput)
                logger.info(requests_per_s)
                if throughput or requests_per_s:
                    logger.info("found throughput and requests_per_s")
                    logger.info({ "throughput": throughput, "requests_per_s": requests_per_s })
                    return { "throughput": throughput, "requests_per_s": requests_per_s }, None
    return {}, f"Throughput and requests_per_s not found for all GPU inference testing"

def get_vram():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    vram_gb = memory_info.total / (1024 ** 3)
    pynvml.nvmlShutdown()

    return math.ceil(vram_gb)

def get_series_name():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device_info = pynvml.nvmlDeviceGetName(handle)
    return device_info

if __name__ == "__main__":
    ans, err = run_multi_gpu_inference(9)
    logger.info(ans)
    logger.error(err)