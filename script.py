import re
import subprocess
import torch
import os
import boto3
from loguru import logger
import json
import time
from collections import defaultdict

os.environ["MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID"] = "AKIAWC2UK4N6Z34M5T6B"
os.environ["MACHINE_BENCHMARK_AWS_SECRET_ACCESS_KEY"] = "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"
os.environ["DEVICE_COUNT"] = "2"
os.environ["MACHINE_BENCHMARK_TOPIC_ARN"] = "arn:aws:sns:us-east-1:418399314813:dev-machine-benchmarks"

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
        logger.info(result.stdout.strip())
        logger.error(result.stderr.strip())
        return result.stdout.strip(), None
    except Exception as e:
        return None, str(e)
    
def download_dataset():
    command = ["wget", "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"]
    return shell_cmd(command)

def setup_dependencies():
    command = ["pip", "install", "-e", "vllm/"]
    return shell_cmd(command)

def setup_axolotl_dependencies():
    command = ["pip", "install", "-e", "axolotl/[flash-attn,deepspeed]"]
    return shell_cmd(command)

def run_lora_on_individual_gpus(device_count):
    errors = []
    outputs = []
    for i in range(device_count):
        command_envs = defaultdict(str)
        command_envs["CUDA_VISIBLE_DEVICES"] = f"{i}"
        command_envs["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        # accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml
        command = ["accelerate", "launch", "-m", "axolotl.cli.train", "axolotl/examples/openllama-3b/lora.yml"]
        out, err = shell_cmd(command, env=command_envs)
        if err != None:
            errors.append(str(err))
        else:
            split_sentences = out.split("\n")
            logger.info(f"last sentence candidate: { split_sentences[-1] }")
            if len(split_sentences) > 0:
                found = False
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
                            outputs.append({ "id": i, "train_runtime": train_runtime, "train_samples_per_second": train_samples_per_second, "train_steps_per_second": train_steps_per_second, "train_loss": train_loss, "epoch": epoch })
                            found = True
                            logger.info("found responses")
                            logger.info({ "train_runtime": train_runtime, "train_samples_per_second": train_samples_per_second, "train_steps_per_second": train_steps_per_second, "train_loss": train_loss, "epoch": epoch })
                if not found:
                    errors.append(f"Could not find benchmarks for finetuning for device: {i}")
    return outputs, "\n".join(errors) if len(errors) > 0 else None

def run_lora_on_all_gpus_together(device_count):
    errors = []
    outputs = []
    command = ["accelerate", "launch", "-m", "axolotl.cli.train", "axolotl/examples/openllama-3b/lora.yml", f"--num_processes={device_count}"]
    out, err = shell_cmd(command)
    if err != None:
        return None, err
    else:
        split_sentences = out.split("\n")
        logger.info(f"last sentence candidate: { split_sentences[-1] }")
        if len(split_sentences) > 0:
            found = False
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
                        outputs.append({ "train_runtime": train_runtime, "train_samples_per_second": train_samples_per_second, "train_steps_per_second": train_steps_per_second, "train_loss": train_loss, "epoch": epoch })
                        found = True
                        logger.info("found responses")
                        logger.info({ "train_runtime": train_runtime, "train_samples_per_second": train_samples_per_second, "train_steps_per_second": train_steps_per_second, "train_loss": train_loss, "epoch": epoch })
            if not found:
                errors.append(f"Could not find benchmarks after finetuning lora on all GPUs.")
    return {}, f"Throughput and requests_per_s not found for all GPU inference testing"

def run_inference_on_individual_gpus(device_count):
    errors = []
    outputs = []
    for i in range(device_count):
        command_envs = defaultdict(str)
        command_envs["CUDA_VISIBLE_DEVICES"] = f"{i}"
        command_envs["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        # python3 vllm/benchmarks/benchmark_throughput.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json
        command = ["python3", "vllm/benchmarks/benchmark_throughput.py", 
                   "--dataset=ShareGPT_V3_unfiltered_cleaned_split.json", "--model=mistralai/Mistral-7B-Instruct-v0.1",
                   "--tokenizer=hf-internal-testing/llama-tokenizer", "--num-prompts=100", "--dtype=bfloat16", "--max-model-len=8192",
                   "--device=cuda"]
        out, err = shell_cmd(command, env=command_envs)
        if err != None:
            errors.append(str(err))
        else:
            split_sentences = out.split("\n")
            logger.info(f"last sentence candidate: { split_sentences[-1] }")
            if len(split_sentences) > 0:
                last_sentence = split_sentences[-1]
                logger.info("last sentence", last_sentence)
                # Regex pattern to extract numbers
                pattern = r"Throughput: ([\d.]+) requests/s, ([\d.]+) tokens/s"
                # Search for matches
                match = re.search(pattern, last_sentence)
                if match:
                    throughput = float(match.group(1))
                    requests_per_s = float(match.group(2))
                    if throughput or requests_per_s:
                        outputs.append({ "gpu": i, "throughput": throughput, "requests_per_s": requests_per_s })
                        logger.info("found throughput and requests_per_s")
                        logger.info({ "id": i, "throughput": throughput, "requests_per_s": requests_per_s })
                else:
                    errors.append(f"Throughput and requests_per_s not found for device: { i }")
    return outputs, "\n".join(errors) if len(errors) > 0 else None

def run_inference_on_all_gpus_together(device_count):
    command_env = defaultdict(str)
    command_env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    command_env["TOKENIZERS_PARALLELISM"] = "false"
    command = ["python3", "vllm/benchmarks/benchmark_throughput.py", 
            "--dataset=ShareGPT_V3_unfiltered_cleaned_split.json", "--model=mistralai/Mistral-7B-Instruct-v0.1",
            "--tokenizer=hf-internal-testing/llama-tokenizer", "--num-prompts=100", "--dtype=bfloat16", 
            "--max-model-len=8192", "--device=cuda", f"--tensor-parallel-size={device_count}"]
    out, err = shell_cmd(command, env=command_env)
    if err != None:
        return None, err
    else:
        split_sentences = out.split("\n")
        if len(split_sentences) > 0:
            logger.info(f"last sentence candidate: { split_sentences[-1] }")
            last_sentence = split_sentences[-1]
            logger.info("last sentence", last_sentence)
            # Regex pattern to extract numbers
            pattern = r"Throughput: ([\d.]+) requests/s, ([\d.]+) tokens/s"
            # Search for matches
            match = re.search(pattern, last_sentence)
            if match:
                throughput = float(match.group(1))
                requests_per_s = float(match.group(2))
                if throughput or requests_per_s:
                    logger.info("found throughput and requests_per_s")
                    logger.info({ "throughput": throughput, "requests_per_s": requests_per_s })
                    return { "throughput": throughput, "requests_per_s": requests_per_s }
            else:
                return None, f"Throughput and requests_per_s not found for all GPU inference testing"
    return {}, f"Throughput and requests_per_s not found for all GPU inference testing"
        

def send_response(client, topic_arn, message):
    message["podId"] = os.environ.get("POD_ID", "missing_pod_id")
    if message.get("errors", None) != None:
        message["errors"] = ";".join(message["errors"])[:300]
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
        expected_device_count = os.environ.get("DEVICE_COUNT", None)
        pod_id = os.environ.get("POD_ID", None)
        aws_key_id =  os.environ.get("MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID", None)
        aws_secret_access_key  = os.environ.get("MACHINE_BENCHMARK_AWS_SECRET_ACCESS_KEY", None)
        aws_topic_arn = os.environ.get("MACHINE_BENCHMARK_TOPIC_ARN", None)
        if not expected_device_count or not pod_id or not aws_key_id or not aws_secret_access_key or not aws_topic_arn:
            logger.debug(f"expected_device_count: { expected_device_count }. pod_id: { pod_id }. aws_key_id: { aws_key_id }. aws_secret_access_key: { aws_secret_access_key }. aws_topic_arn: { aws_topic_arn }")
            if not aws_key_id or not aws_secret_access_key or not aws_topic_arn or not pod_id:
                logger.error("""One of the critical environment variables is missing: MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID, 
                                MACHINE_BENCHMARK_AWS_SECRET_ACCESS_KEY, MACHINE_BENCHMARK_TOPIC_ARN. 
                                Cannot send request to terminate the pod.""")
                return
            else:
                client = initialize_aws_client(aws_key_id, aws_secret_access_key, "us-east-1")
                send_response(client, aws_topic_arn, {"errors": ["""One of the environment variables is missing: 
                                        POD_ID, MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID, 
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

        setup_dependencies_out, setup_dependencies_errors = setup_dependencies()
        if setup_dependencies_errors != None:
            logger.error(setup_dependencies_errors)
            send_response(client, aws_topic_arn, {"errors": [setup_dependencies_errors]})
            return
        
        separate_gpus_inference_output, separate_gpus_inference_errors = run_inference_on_individual_gpus(device_count)
        if separate_gpus_inference_errors != None:
            logger.error(separate_gpus_inference_errors)
            send_response(client, aws_topic_arn, {"errors": [separate_gpus_inference_errors]})
            return

        combined_gpus_inference_output, combined_gpus_inference_errors = run_inference_on_all_gpus_together(device_count)
        if combined_gpus_inference_errors != None:
            logger.error(combined_gpus_inference_errors)
            send_response(client, aws_topic_arn, {"errors": [combined_gpus_inference_errors]})
            return
        ### inference complete

        ### training
        separate_gpus_finetuning_output, separate_gpus_finetuning_errors = run_lora_on_individual_gpus(device_count)
        if separate_gpus_finetuning_errors != None:
            logger.error(separate_gpus_finetuning_errors)
            send_response(client, aws_topic_arn, {"errors": [separate_gpus_finetuning_errors]})
            return
        
        combined_gpus_finetuning_output, combined_gpus_finetuning_errors = run_lora_on_all_gpus_together(device_count)
        if combined_gpus_finetuning_errors != None:
            logger.error(combined_gpus_finetuning_errors)
            send_response(client, aws_topic_arn, {"errors": [combined_gpus_finetuning_errors]})
            return
        
        payload = {
            "separate_gpus_inference_output": separate_gpus_inference_output,
            "combined_gpus_inference_output": combined_gpus_inference_output,
            "separate_gpus_finetuning_output": separate_gpus_finetuning_output,
            "combined_gpus_finetuning_output": combined_gpus_finetuning_output,
        }
        send_response(client, aws_topic_arn, payload)
    except Exception as e:
        logger.exception(f"Expected {str(e)}")
        send_response(client, aws_topic_arn, {"errors": [str(e)]})

if __name__ == "__main__":
    os.environ["MACHINE_BENCHMARK_AWS_ACCESS_KEY_ID"] = "AKIAWC2UK4N6Z34M5T6B"
    os.environ["MACHINE_BENCHMARK_AWS_SECRET_ACCESS_KEY"] = "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"
    os.environ["DEVICE_COUNT"] = "2"
    os.environ["MACHINE_BENCHMARK_TOPIC_ARN"] = "arn:aws:sns:us-east-1:418399314813:dev-machine-benchmarks"
    main()


# TODO:
    # export HF_HUB_ENABLE_HF_TRANSFER=1
    # pip install hf-transfer

    # --max-model-len=8192
    # reduce samples
    # export TOKENIZERS_PARALLELISM=false