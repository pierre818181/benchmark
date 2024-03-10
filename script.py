import re
import subprocess
import torch
import os
import boto3
from loguru import logger
import json
import time
from collections import defaultdict
import pynvml
import math

def shell_cmd(command, env={}):
    try:
        global_envs = os.environ.copy()
        if env:
            for k, v in env.items():
                global_envs[k] = v
        logger.info(f"running command: { ' '.join(command) }")
        logger.info(f"env vars: {env}")
        result = subprocess.run(command, capture_output=True, text=True, env=global_envs)

        if result.returncode != 0:
            logger.error(result.stderr.strip())
            logger.error(f"could not execute shell command: { ' '.join(command) }")
            return None, result.stderr.strip()

        if os.environ.get("VERBOSE", "true") == "true":
            logger.info("Command output:")
            logger.info(result.stdout.strip())
            logger.error(result.stderr.strip())
        return result.stdout.strip(), None
    except Exception as e:
        return None, str(e)
    
def download_dataset():
    command = ["wget", "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"]
    return shell_cmd(command)

def setup_inference_dependencies():
    command = ["pip", "install", "-e", "vllm/"]
    return shell_cmd(command)

def setup_finetune_dependencies():
    command = ["pip", "install", "-e", "axolotl/[flash-attn,deepspeed]"]
    return shell_cmd(command)

def run_single_shell_cmd(command):
    res = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return None, res.stderr.strip()
    return res.stdout.strip(), None

def adjust_max_steps():
    out, err = run_single_shell_cmd(f'echo "\nmax_steps: 30" | tee -a axolotl/examples/mistral/config.yml > /dev/null')
    if err != None:
        return err
    out, err = run_single_shell_cmd(f'echo "\nmax_steps: 30" | tee -a axolotl/examples/openllama-3b/lora.yml > /dev/null')
    if err != None:
        return err
    out, err = run_single_shell_cmd(f'echo "\nmax_steps: 30" | tee -a axolotl/examples/openllama-3b/config.yml > /dev/null')
    if err != None:
        return err
    out, err = run_single_shell_cmd(f'echo "\nmax_steps: 30" | tee -a axolotl/examples/tiny-llama/qlora.yml > /dev/null')
    if err != None:
        return err
    # eval_sample_packing: False
    out, err = run_single_shell_cmd(f'echo "\neval_sample_packing: False" | tee -a axolotl/examples/tiny-llama/qlora.yml > /dev/null')
    if err != None:
        return err
    out, err = run_single_shell_cmd(f"sed -i 's/micro_batch_size: 2/micro_batch_size: 1/' axolotl/examples/tiny-llama/qlora.yml")
    if err != None:
        return err
    out, err = run_single_shell_cmd(f"sed -i 's/micro_batch_size: 2/micro_batch_size: 1/' axolotl/examples/openllama-3b/lora.yml")
    if err != None:
        return err
    out, err = run_single_shell_cmd(f"sed -i 's/micro_batch_size: 2/micro_batch_size: 1/' axolotl/examples/openllama-3b/config.yml")
    if err != None:
        return err
    return None

def get_vram():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    vram_gb = memory_info.total / (1024 ** 3)
    pynvml.nvmlShutdown()

    return math.ceil(vram_gb)

def get_gpu_series_name():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device_info = pynvml.nvmlDeviceGetName(handle)
    return device_info

def get_finetune_config(vram):
    if vram >= 32:
        return "axolotl/examples/openllama-3b/config.yml"
    elif vram >= 8:
        return "axolotl/examples/openllama-3b/lora.yml"
    else:
        return "axolotl/examples/tiny-llama/qlora.yml"

def run_single_gpu_finetune(device_count):
    errors = []
    outputs = []
    vram = get_vram()
    gpu_series = get_gpu_series_name()
    config = get_finetune_config(vram)
    for i in range(device_count):
        command_envs = defaultdict(str)
        command_envs["CUDA_VISIBLE_DEVICES"] = f"{i}"
        command_envs["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        if "V100" in gpu_series:
            out, err = subprocess.run(f"sed -i 's/flash_attention: true/flash_attention: false/' {config}", shell=True)
            if err != None:
                return out, err
        command = ["accelerate", "launch", "-m", "axolotl.cli.train", config,]
        out, err = shell_cmd(command, env=command_envs)
        if err != None:
            errors.append(str(err))
        else:
            split_sentences = out.split("\n")
            if len(split_sentences) > 0:
                found = False
                for sentence in split_sentences:
                    updated_pattern = r"{'train_runtime': ([\d.]+), 'train_samples_per_second': ([\d.]+), 'train_steps_per_second': ([\d.]+), 'train_loss': ([\d.]+), 'epoch': ([\d.]+)}"
                    match = re.search(updated_pattern, sentence)
                    if match:
                        logger.info(f"matched sentence: { sentence }")
                        train_runtime = float(match.group(1))
                        train_samples_per_second = float(match.group(2))
                        train_steps_per_second = float(match.group(3))
                        train_loss = float(match.group(4))
                        epoch = float(match.group(5))
                        if train_runtime or train_samples_per_second or train_steps_per_second or train_loss or epoch:
                            outputs.append({ "id": i, "train_runtime": train_runtime, "train_samples_per_second": train_samples_per_second, "train_steps_per_second": train_steps_per_second, "train_loss": train_loss, "epoch": epoch })
                            found = True
                            logger.info("found responses")
                            logger.info({ "id": i, "train_runtime": train_runtime, "train_samples_per_second": train_samples_per_second, "train_steps_per_second": train_steps_per_second, "train_loss": train_loss, "epoch": epoch })
                if not found:
                    errors.append(f"Could not find benchmarks for finetuning for device: {i}")
    return outputs, "\n".join(errors) if len(errors) > 0 else None

# Currently using a small model to run on all GPUs. In the future, we might want to change to bigger models.
def run_multi_gpu_finetune(device_count):
    total_vram = get_vram()
    config = get_finetune_config(total_vram)
    command = ["accelerate", "launch", "-m", "axolotl.cli.train", config]
    out, err = shell_cmd(command)
    if err != None:
        return None, err
    else:
        split_sentences = out.split("\n")
        for sentence in split_sentences:
            updated_pattern = r"{'train_runtime': ([\d.]+), 'train_samples_per_second': ([\d.]+), 'train_steps_per_second': ([\d.]+), 'train_loss': ([\d.]+), 'epoch': ([\d.]+)}"
            match = re.search(updated_pattern, sentence)
            if match:
                logger.info("last sentence", sentence)
                train_runtime = float(match.group(1))
                train_samples_per_second = float(match.group(2))
                train_steps_per_second = float(match.group(3))
                train_loss = float(match.group(4))
                epoch = float(match.group(5))
                if train_runtime or train_samples_per_second or train_steps_per_second or train_loss or epoch:
                    logger.info({ "train_runtime": train_runtime, "train_samples_per_second": train_samples_per_second, "train_steps_per_second": train_steps_per_second, "train_loss": train_loss, "epoch": epoch })
                    return { "train_runtime": train_runtime, "train_samples_per_second": train_samples_per_second, "train_steps_per_second": train_steps_per_second, "train_loss": train_loss, "epoch": epoch }, None
    return {}, f"Throughput and requests_per_s not found for all GPU finetune testing"

def get_parallel_size_accounting_for_attn_head(device_count):
    possible_parallel_size = [32,16,8,4,2,1]
    for p_size in possible_parallel_size:
        if device_count >= p_size:
            return p_size
    
    return 1

def run_single_gpu_inference(device_count):
    errors = []
    outputs = []
    vram = get_vram()
    model = "mistralai/Mistral-7B-Instruct-v0.1" if vram >= 12 else "facebook/opt-125m"
    max_model_len = 3152 if vram >= 12 else 2048
    gpu_series_name = get_gpu_series_name()
    for i in range(device_count):
        command_envs = defaultdict(str)
        command_envs["CUDA_VISIBLE_DEVICES"] = f"{i}"
        command_envs["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        command = ["python3", "vllm/benchmarks/benchmark_throughput.py", 
                   "--dataset=ShareGPT_V3_unfiltered_cleaned_split.json", f"--model={model}",
                    "--num-prompts=30", f"--max-model-len={max_model_len}",
                   "--device=cuda", "--enforce-eager", "--gpu-memory-utilization=0.95"]
        if vram >= 12:
            command.append("--tokenizer=hf-internal-testing/llama-tokenizer")
        if "V100" in gpu_series_name:
            command.append("--dtype=float16")
        else:
            command.append("--dtype=auto")
        out, err = shell_cmd(command, env=command_envs)
        if err != None:
            errors.append(str(err))
        else:
            found = False
            split_sentences = out.split("\n")
            for sentence in split_sentences:
                pattern = r"Throughput: ([\d.]+) requests/s, ([\d.]+) tokens/s"
                match = re.search(pattern, sentence)
                if match:
                    found = True
                    logger.info(f"matched sentence: { sentence }")
                    throughput = float(match.group(1))
                    requests_per_s = float(match.group(2))
                    if throughput or requests_per_s:
                        outputs.append({ "id": i, "throughput": throughput, "requests_per_s": requests_per_s })
                        logger.info("found throughput and requests_per_s")
                        logger.info({ "id": i, "throughput": throughput, "requests_per_s": requests_per_s })
            if not found:
                errors.append(f"Throughput and requests_per_s not found for device: { i }")
    return outputs, "\n".join(errors) if len(errors) > 0 else None

def run_multi_gpu_inference(device_count):
    command_env = defaultdict(str)
    command_env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    command_env["TOKENIZERS_PARALLELISM"] = "false"
    vram = get_vram()
    model = "mistralai/Mistral-7B-Instruct-v0.1" if vram >= 12 else "facebook/opt-125m"
    device_count_attn_heads = get_parallel_size_accounting_for_attn_head(device_count)
    max_model_len = 3152 * device_count_attn_heads if vram >= 12 else 2048
    gpu_series_name = get_gpu_series_name()
    if device_count > 32:
        return None, f"Inference for devices with more than 32 GPUs not supported at once. Current GPU count: { device_count }"
    command = ["python3", "vllm/benchmarks/benchmark_throughput.py", 
            "--dataset=ShareGPT_V3_unfiltered_cleaned_split.json", f"--model={model}",
            "--num-prompts=30", f"--max-model-len={ max_model_len }", "--device=cuda", 
            f"--tensor-parallel-size={device_count_attn_heads}", "--enforce-eager", "--gpu-memory-utilization=0.95"]
    if vram >= 12:
        command.append("--tokenizer=hf-internal-testing/llama-tokenizer")
    if "V100" in gpu_series_name:
        command.append("--dtype=float16")
    else:
        command.append("--dtype=auto")
    out, err = shell_cmd(command, env=command_env)
    if err != None:
        return None, err
    else:
        split_sentences = out.split("\n")
        for sentence in split_sentences:
            # Regex pattern to extract numbers
            pattern = r"Throughput: ([\d.]+) requests/s, ([\d.]+) tokens/s"
            # Search for matches
            match = re.search(pattern, sentence)
            if match:
                logger.info("matched")
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
        

def send_response(client, topic_arn, message):
    message["podId"] = os.environ.get("RUNPOD_POD_ID", "missing_pod_id")
    if message.get("errors", None) != None:
        message["errors"] = ";".join(message["errors"])[-3000:]
    
    logger.info("sending response")
    logger.info(json.dumps(message))
    logger.info(message)
    response = client.publish(
        TopicArn=topic_arn,
        Message=json.dumps(message)
    )

    logger.info(f"response from sns: {response}")
    time.sleep(10)

def initialize_aws_client(access_key, secret_key, region):
    return boto3.client(
        'sns',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

def main():
    try:
        expected_device_count = os.environ.get("RUNPOD_GPU_COUNT", None)
        pod_id = os.environ.get("RUNPOD_POD_ID", None)
        aws_key_id = os.environ.get("MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID", None)
        aws_secret_access_key  = os.environ.get("MACHINE_BENCHMARK_AWS_SECRET_ACCESS_KEY", None)
        aws_topic_arn = os.environ.get("MACHINE_BENCHMARK_TOPIC_ARN", None)
        if not expected_device_count or not pod_id or not aws_key_id or not aws_secret_access_key or not aws_topic_arn:
            logger.info(f"expected_device_count: { expected_device_count }. pod_id: { pod_id }. aws_key_id: { aws_key_id }. aws_secret_access_key: { aws_secret_access_key }. aws_topic_arn: { aws_topic_arn }")
            if not aws_key_id or not aws_secret_access_key or not aws_topic_arn or not pod_id:
                logger.error("""One of the critical environment variables is missing: MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID, 
                                MACHINE_BENCHMARK_AWS_SECRET_ACCESS_KEY, MACHINE_BENCHMARK_TOPIC_ARN. 
                                Cannot send request to terminate the pod. Please terminate the pod manually.""")
                return
            else:
                client = initialize_aws_client(aws_key_id, aws_secret_access_key, "us-east-1")
                send_response(client, aws_topic_arn, {"errors": ["""One of the environment variables is missing: 
                                        RUNPOD_POD_ID, RUNPOD_GPU_COUNT, MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID, 
                                        MACHINE_BENCHMARK_AWS_SECRET_ACCESS_KEY, MACHINE_BENCHMARK_TOPIC_ARN"""],
                                    })
                return
        client = initialize_aws_client(aws_key_id, aws_secret_access_key, "us-east-1")

        device_count = torch.cuda.device_count()
        if device_count != int(expected_device_count):
            logger.error(f"Expected {expected_device_count} GPUs, but found {device_count} GPUs")
            send_response(client, aws_topic_arn, {"errors": [f"Expected {expected_device_count} GPUs, but found {device_count} GPUs"]})
            return

        download_dataset_out, download_dataset_errors = download_dataset()
        if download_dataset_errors != None:
            logger.error(download_dataset_errors)
            send_response(client, aws_topic_arn, {"errors": [download_dataset_errors]})
            return

        payload = {}
        setup_dependencies_out, setup_dependencies_errors = setup_inference_dependencies()
        if setup_dependencies_errors != None:
            logger.info("errored during inference dependencies installation")
            logger.error(setup_dependencies_errors)
            send_response(client, aws_topic_arn, {"errors": [setup_dependencies_errors]})
            return

        ## inference starts
        if os.environ.get("RUN_INFERENCE", "true") == "true":
            if os.environ.get("RUN_MULTI_INFERENCE", "true") == "true":
                multi_gpu_inference_output, multi_gpu_inference_errors = run_multi_gpu_inference(device_count)
                if multi_gpu_inference_errors != None:
                    logger.info("errored during multi gpu inference")
                    logger.exception(multi_gpu_inference_errors)
                    payload["errors"] = [multi_gpu_inference_errors]
                    send_response(client, aws_topic_arn, payload)
                    return
                payload["multiGpuInferenceResult"] = multi_gpu_inference_output
            
            if os.environ.get("RUN_ISOLATED_INFERENCE", "true") == "true":
                single_gpu_inference_outputs, single_gpu_inference_errors = run_single_gpu_inference(device_count)
                if single_gpu_inference_errors != None:
                    logger.info("errored during single gpu inference")
                    logger.exception(single_gpu_inference_errors)
                    payload["errors"] = [single_gpu_inference_errors]
                    send_response(client, aws_topic_arn, payload)
                    return
                payload["singleGpuInferenceResults"] = single_gpu_inference_outputs
        ## inference ends

        ### training
        if os.environ.get("RUN_FINETUNE", "true") == "true":
            setup_lora_dependencies_output, setup_finetune_dependencies_errors = setup_finetune_dependencies()
            if setup_finetune_dependencies_errors != None:
                logger.info("errored during finetune dependencies installation")
                logger.error(setup_finetune_dependencies_errors)
                payload["errors"] = [setup_finetune_dependencies_errors]
                send_response(client, aws_topic_arn, payload)
                return
            
            adjust_max_steps_err = adjust_max_steps()
            if adjust_max_steps_err != None:
                logger.info("errored when adjusting max steps")
                logger.error(adjust_max_steps_err)
                payload["errors"] = [adjust_max_steps_err]
                send_response(client, aws_topic_arn, payload)
                return
            
            if os.environ.get("RUN_MULTI_FINETUNE", "true") == "true":
                multi_gpu_finetune_output, multi_gpu_finetune_errors = run_multi_gpu_finetune(device_count)
                if multi_gpu_finetune_errors != None:
                    logger.info("errored during multi gpu finetune")
                    logger.error(multi_gpu_finetune_errors)
                    payload["errors"] = [multi_gpu_finetune_errors]
                    send_response(client, aws_topic_arn, payload)
                    return
                payload["multiGpuFinetuneResults"] = multi_gpu_finetune_output

            if os.environ.get("RUN_ISOLATED_FINETUNE", "true") == "true":
                single_gpu_finetune_outputs, single_gpu_finetune_errors = run_single_gpu_finetune(device_count)
                if single_gpu_finetune_errors != None:
                    logger.info("errored during single gpu finetune")
                    logger.error(single_gpu_finetune_errors)
                    payload["errors"] = [single_gpu_finetune_errors]
                    send_response(client, aws_topic_arn, payload)
                    return
                payload["singleGpuFinetuneResults"] = single_gpu_finetune_outputs
        ### training ends

        logger.info("sending payload", payload)
        send_response(client, aws_topic_arn, payload)
    except Exception as e:
        logger.exception(f"Exception {str(e)}")
        send_response(client, aws_topic_arn, {"errors": [str(e)]})

if __name__ == "__main__":
    main()