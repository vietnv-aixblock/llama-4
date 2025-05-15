#model_marketplace.config
# {"token_length": "4018", "accuracy": "70", "precision": "fp16", "sampling_frequency:": "44100", "mono": true, "fps": "74", "resolution": "480", "image_width": "1080", "image_height": "1920", "framework": "transformers", "dataset_format": "llm", "dataset_sample": "[id on s3]", "weights": [
#     {
#       "name": "DeepSeek-V3",
#       "value": "deepseek-ai/DeepSeek-V3",
#       "size": 20,
#       "paramasters": "685B",
#       "tflops": 14, 
#       "vram": 20,
#       "nodes": 10
#     },
# {
#       "name": "DeepSeek-V3-bf16",
#       "value": "opensourcerelease/DeepSeek-V3-bf16",
#       "size": 1500,
#       "paramasters": "684B",
#       "tflops": 80, 
#       "vram": 48,
#       "nodes": 10
#     }
#   ], "cuda": "11.4", "task":["text-generation", "text-classification", "text-summarization", "text-ner", "question-answering"]}
import math
import pathlib
import pickle
import random
import subprocess
import time
from typing import List, Dict, Optional
import accelerate
from aixblock_ml.model import AIxBlockMLBase
import numpy as np
import pandas as pd
import torch
from transformers import pipeline
import os
import zipfile
from huggingface_hub import HfFolder
import wandb
from prompt import qa_with_context, text_classification, text_summarization, qa_without_context,text_ner, chatbot_with_history
from logging_class import start_queue, write_log

# Đặt token của bạn vào đây
hf_token = os.getenv("HF_TOKEN", "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")
# Lưu token vào local
HfFolder.save_token(hf_token)
# wandb.login('allow',"69b9681e7dc41d211e8c93a3ba9a6fb8d781404a")
# print("Login successful")
from huggingface_hub import login
hf_access_token = "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"
login(token = hf_access_token)

if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    print("CUDA is available.")

    _model = pipeline(
        "text-generation",
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct", #"meta-llama/Llama-4-Scout-17B-16E-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
        torch_dtype=dtype,
        device_map="auto",  # Hoặc có thể thử "cpu" nếu không ổn,
        max_new_tokens=256,
        token = "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"
    )
else:
    print("No GPU available, using CPU.")
    _model = pipeline(
        "text-generation",
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct", #"meta-llama/Llama-4-Scout-17B-16E-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
        device_map="cpu",
        max_new_tokens=256,
        token = "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"
    )


from typing import List, Dict, Optional
from aixblock_ml.model import AIxBlockMLBase
import torch.distributed as dist
import os
import torch
import os
import subprocess
import random
import logging
import subprocess
import time
import json


HOST_NAME = os.environ.get('HOST_NAME',"https://dev-us-west-1.aixblock.io")
# HOST_NAME = os.environ.get('HOST_NAME',"http://127.0.0.1:8080")
TYPE_ENV = os.environ.get('TYPE_ENV',"DETECTION")
import requests
from function_ml import connect_project, download_dataset, upload_checkpoint

# def download_dataset(data_zip_dir, project_id, dataset_id, token):
#     url = f"{HOST_NAME}/api/dataset_model_marketplace/download/{dataset_id}?project_id={project_id}"
#     payload = {}
#     headers = {
#         'accept': 'application/json',
#         'Authorization': f'Token {token}'
#     }

#     response = requests.request("GET", url, headers=headers, data=payload)
#     dataset_name = response.headers.get('X-Dataset-Name')
#     if response.status_code == 200:
#         with open(data_zip_dir, 'wb') as f:
#             f.write(response.content)
#         return dataset_name
#     else:
#         return None

# def upload_checkpoint(checkpoint_model_dir, project_id, token, path_file, send_mail=False):
#     import os
#     url = f"{HOST_NAME}/api/checkpoint_model_marketplace/upload/"

#     payload = {
#         "type_checkpoint": "ml_checkpoint",
#         "project_id": f'{project_id}',
#         "is_training": send_mail,
#         "full_path": path_file
#     }
#     headers = {
#         'accept': 'application/json',
#         'Authorization': f'Token {token}'
#     }

#     checkpoint_name = None

#     # response = requests.request("POST", url, headers=headers, data=payload) 
#     with open(checkpoint_model_dir, 'rb') as file:
#         files = {'file': file}
#         response = requests.post(url, headers=headers, files=files, data=payload)
#         checkpoint_name = response.headers.get('X-Checkpoint-Name')

#     print("upload done")
#     return checkpoint_name

# def read_dataset(file_path):
#     # Kiểm tra xem thư mục /content/ có tồn tại không
#     if os.path.isdir(file_path):
#         files = os.listdir(file_path)
#         # Kiểm tra xem có file json nào không
#         for file in files:
#             if file.endswith(".json"):
#             # Đọc file json
#                 with open(os.path.join(file_path, file), "r") as f:
#                     data = json.load(f)

#                 return data
#     return None

# def is_correct_format(data_json):
#     try:
#         for item in data_json:
#             if not all(key in item for key in ['instruction', 'input', 'output']):
#                 return False
#         return True
#     except Exception as e:
#         return False

# def conver_to_hf_dataset(data_json):
#     formatted_data = []
#     for item in data_json:
#         for annotation in item['annotations']:
#             question = None
#             answer = None
#             for result in annotation['result']:
#                 if result['from_name'] == 'question':
#                     question = result['value']['text'][0]
#                 elif result['from_name'] == 'answer':
#                     answer = result['value']['text'][0]
#             if question and answer:
#                 formatted_data.append({
#                     'instruction': item['data']['text'],
#                     'input': question,
#                     'output': answer
#                 })
#     return formatted_data

class MyModel(AIxBlockMLBase):

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ 
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}''')
        return []

    def fit(self, event, data, **kwargs):
        """

        
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')
    def action(self, command, **kwargs):

        print(f"""
         
                command: {command},
              """)
        if command.lower() == "train":
                import threading
                import os

                model_id = kwargs.get("model_id", "bigscience/bloomz-1b7")  #"tiiuae/falcon-7b" "bigscience/bloomz-1b7" `zanchat/falcon-1b` `appvoid/llama-3-1b` meta-llama/Llama-3.2-3B` `mistralai/Mistral-7B-v0.1` `bigscience/bloomz-1b7` `Qwen/Qwen2-1.5B`
                dataset_id = kwargs.get("dataset_id","lucasmccabe-lmi/CodeAlpaca-20k") #gingdev/llama_vi_52k kigner/ruozhiba-llama3-tt

                push_to_hub = kwargs.get("push_to_hub", False)
                hf_model_id = kwargs.get("hf_model_id", "llama3")
                push_to_hub_token = kwargs.get("push_to_hub_token", "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")

                task = kwargs.get("task", "text-generation")
                framework = kwargs.get("framework", "huggingface")
                prompt = kwargs.get("prompt", "")
                report_to = kwargs.get("report_to", "tensorboard")
                wantdb_api_key = kwargs.get("wantdb_api_key", "69b9681e7dc41d211e8c93a3ba9a6fb8d781404a")
                trainingArguments = kwargs.get("TrainingArguments", None)

                json_file = "training_args.json"
                absolute_path = os.path.abspath(json_file)
                action = "train"

                with open(absolute_path, 'w') as f:
                    json.dump(trainingArguments, f)

                print(trainingArguments)
                if len(wantdb_api_key)> 0 and wantdb_api_key != "69b9681e7dc41d211e8c93a3ba9a6fb8d781404a":
                    wandb.login('allow',wantdb_api_key)

                os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
                clone_dir = os.path.join(os.getcwd())
                # epochs = kwargs.get("num_epochs", 10)
                project_id = kwargs.get("project_id")
                token = kwargs.get("token","ebcf0ceda01518700f41dfa234b6f4aaea0b57af")
                checkpoint_version = kwargs.get("checkpoint_version")
                checkpoint_id = kwargs.get("checkpoint")
                dataset_version = kwargs.get("dataset_version")
                dataset_id = kwargs.get("dataset")
                channel_log = kwargs.get("channel_log", "training_logs")
                world_size = kwargs.get("world_size", 1)
                rank = kwargs.get("rank", 0)
                master_add = kwargs.get("master_add","127.0.0.1")
                master_port = kwargs.get("master_port", "23456")
                # entry_file = kwargs.get("entry_file")
                configs = kwargs.get("configs")
                host_name = kwargs.get("host_name",HOST_NAME)
                log_queue, logging_thread = start_queue(channel_log)
                write_log(log_queue)
                # hyperparameter = kwargs.get("hyperparameter")

                # !pip install nvidia-ml-py
                # from pynvml import *
                # nvmlInit()
                # print(f"Driver Version: {nvmlSystemGetDriverVersion()}")
                # # Driver Version: 11.515.48
                # deviceCount = nvmlDeviceGetCount()
                # for i in range(deviceCount):
                # handle = nvmlDeviceGetHandleByIndex(i)
                # print(f"Device {i} : {nvmlDeviceGetName(handle)}")
                # nvmlShutdown()

                def func_train_model(clone_dir, project_id, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id,model_id,world_size,rank,master_add,master_port,prompt, json_file, channel_log, hf_model_id, push_to_hub, push_to_hub_token,host_name):
                    # from misc import get_device_counts
                    import os
                    dataset_path = None
                    project = connect_project(host_name, token, project_id)
                    print("Connect project:", project)

                    if dataset_version and dataset_id:
                        dataset_path = os.path.join(clone_dir, f"datasets/{dataset_version}")

                        if not os.path.exists(dataset_path):
                            data_path = os.path.join(clone_dir, "data_zip")
                            os.makedirs(data_path, exist_ok=True)
                            # data_zip_dir = os.path.join(clone_dir, "data_zip/data.zip")
                            # dataset_name = download_dataset(data_zip_dir, project_id, dataset_id, token)
                            dataset_name = download_dataset(project, dataset_id, data_path)
                            print(dataset_name)
                            if dataset_name:
                                data_zip_dir = os.path.join(data_path, dataset_name)
                                # if not os.path.exists(dataset_path):
                                with zipfile.ZipFile(data_zip_dir, 'r') as zip_ref:
                                    zip_ref.extractall(dataset_path)

                    import torch
                    # https://huggingface.co/docs/accelerate/en/basic_tutorials/launch
                    # https://huggingface.co/docs/accelerate/en/package_reference/cli
                    subprocess.run(
                                (
                                    "whereis accelerate"
                                ),
                                shell=True,
                            )
                    # https://github.com/huggingface/accelerate/issues/1474
                    if framework == "huggingface":
                        if int(world_size) > 1:
                            if int(rank) == 0:
                                print("master node")
                                #  --dynamo_backend 'no' \
                                # --rdzv_backend c10d
                                command = (
                                        "accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank 0 --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action={action}"
                                    ).format(
                                        num_processes=world_size*torch.cuda.device_count(),
                                        SLURM_NNODES=world_size,
                                        head_node_ip=master_add,
                                        port=master_port,
                                        file_name="./run_distributed_accelerate.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                            else:
                                print("worker node")
                                # --rdzv_backend c10d
                                command = (
                                        "accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank {machine_rank} --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action={action}"
                                    ).format(
                                        num_processes=world_size*torch.cuda.device_count(),
                                        SLURM_NNODES=world_size,
                                        head_node_ip=master_add,
                                        port=master_port,
                                        machine_rank=rank,
                                        file_name="./run_distributed_accelerate.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                        else:
                            if torch.cuda.device_count() > 1: # multi gpu
                                            #                                     --rdzv_backend c10d \
                                            # --main_process_ip {head_node_ip} \
                                            # --main_process_port {port} \
                                            # --mixed_precision 'fp16' \
                                            # --num_machines {SLURM_NNODES} \
                                            # --num_processes {num_processes}
                                command = (
                                        "accelerate launch --multi_gpu --num_machines {SLURM_NNODES} --machine_rank 0 --num_processes {num_processes} {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                    ).format(
                                        num_processes=world_size*torch.cuda.device_count(),
                                        SLURM_NNODES=world_size,
                                        # head_node_ip=os.environ.get("head_node_ip", master_add),
                                        port=master_port,
                                        file_name="./run_distributed_accelerate.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                                # mixed_precision: fp16
                            elif torch.cuda.device_count() == 1: # one gpu
                                # num_processes = world_size*get_device_count()
                                #    --rdzv_backend c10d \
                                #     --main_process_ip {head_node_ip} \
                                #     --main_process_port {port} \
                                # --num_cpu_threads_per_process=2 --num_processes {num_processes} \
                                #     --num_machines {SLURM_NNODES} \
                                command = (
                                    "accelerate launch {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action={action}"
                                ).format(
                                    # num_processes=os.environ.get("num_processes", num_processes),
                                    # SLURM_NNODES=os.environ.get("SLURM_NNODES", world_size),
                                    # head_node_ip=os.environ.get("head_node_ip", master_add),
                                    # port=os.environ.get("port", master_port),
                                    file_name="./run_distributed_accelerate.py",
                                    json_file=json_file,
                                    dataset_path=dataset_path,
                                    channel_log=channel_log,
                                    hf_model_id=hf_model_id,
                                    push_to_hub=push_to_hub,
                                    push_to_hub_token=push_to_hub_token,
                                    action=action
                                )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                                # python3.10 -m pip install -r requirements.txt
                            else: # no gpu
                                command = (
                                        "accelerate launch --cpu {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action={action}"
                                    ).format(
                                        file_name="./run_distributed_accelerate.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                print(command)
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)

                    elif framework == "pytorch":
                        import torch
                        # from peft import AutoPeftModelForCausalLM
                        # from transformers import AutoTokenizer, pipeline
                        # from datasets import load_dataset
                        # from random import randint
                        # from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments #BitsAndBytesConfig,AutoConfig,
                        # from trl import SFTTrainer
                        # import pathlib
                        # from transformers import TrainerCallback
                        # import numpy as np
                        # from torch.utils.data.dataloader import DataLoader
                        # from transformers import DataCollatorWithPadding
                        process = subprocess.run(
                                (
                                    "whereis torchrun"
                                ),
                                shell=True,
                            )
                        # print(process.stdout.replace("torchrun: ",""))
                        if int(world_size) > 1:
                            if rank == 0:
                                print("master node")
                                # args = f"--model_id {model_id} --dataset_id {dataset_id} --num_train_epochs {num_train_epochs} " \
                                #     f"--bf16 {bf16} --fp16 {fp16} --use_cpu {use_cpu} --push_to_hub {push_to_hub} " \
                                #     f"--hf_model_id {hf_model_id} --max_seq_length {max_seq_length} --framework {framework} " \
                                #     f"--per_device_train_batch_size {per_device_train_batch_size} --gradient_accumulation_steps {gradient_accumulation_steps} " \
                                #     f"--gradient_checkpointing {gradient_checkpointing} --optim {optim} --logging_steps {logging_steps} " \
                                #     f"--learning_rate {learning_rate} --max_grad_norm {max_grad_norm} --lora_alpha {lora_alpha} " \
                                #     f"--lora_dropout {lora_dropout} --bias {bias} --target_modules {target_modules} --task_type {task_type}"
                                #  --dynamo_backend 'no' \
                                command = (
                                        "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                        "--master_addr {master_addr} --master_port {master_port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                    ).format(
                                         nnodes=int(world_size),
                                        node_rank= int(rank),
                                        nproc_per_node=world_size*torch.cuda.device_count(),
                                        master_addr="127.0.0.1",
                                        master_port="23456",
                                        file_name="./run_distributed_pytorch.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                            else:
                                print("worker node")
                                # args = f"--model_id {model_id} --dataset_id {dataset_id} --num_train_epochs {num_train_epochs} " \
                                #     f"--bf16 {bf16} --fp16 {fp16} --use_cpu {use_cpu} --push_to_hub {push_to_hub} " \
                                #     f"--hf_model_id {hf_model_id} --max_seq_length {max_seq_length} --framework {framework} " \
                                #     f"--per_device_train_batch_size {per_device_train_batch_size} --gradient_accumulation_steps {gradient_accumulation_steps} " \
                                #     f"--gradient_checkpointing {gradient_checkpointing} --optim {optim} --logging_steps {logging_steps} " \
                                #     f"--learning_rate {learning_rate} --max_grad_norm {max_grad_norm} --lora_alpha {lora_alpha} " \
                                #     f"--lora_dropout {lora_dropout} --bias {bias} --target_modules {target_modules} --task_type {task_type}"
                                command = (
                                        "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                        "--master_addr {master_addr} --master_port {master_port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                    ).format(
                                        nnodes=int(world_size),
                                        node_rank= int(rank),
                                        nproc_per_node=world_size*torch.cuda.device_count(),
                                        master_addr=master_add,
                                        master_port=master_port,
                                        file_name="./run_distributed_pytorch.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                print(command)
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                        else:
                            command = (
                                        "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                        "{file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                    ).format(
                                        nnodes=int(world_size),
                                        node_rank= int(rank),
                                        nproc_per_node=world_size*torch.cuda.device_count(),
                                        # master_addr=master_add,
                                        # master_port=master_port,
                                        file_name="./run_distributed_pytorch.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                            process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )

                                # run_train(command)
                    elif framework == "sglang":
                        # https://docs.sglang.ai/
                        # !python3 -m sglang.launch_server --model tiiuae/Falcon3-1B-Instruct --trust-remote-code --tp 1 --port 30000 --host 0.0.0.0
                        if int(world_size) > 1:
                            if int(rank) == 0:
                                print("master node")
                                #  --dynamo_backend 'no' \
                                # --rdzv_backend c10d
                                command = (
                                        "accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank 0 --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                    ).format(
                                        num_processes=world_size*torch.cuda.device_count(),
                                        SLURM_NNODES=world_size,
                                        head_node_ip=master_add,
                                        port=master_port,
                                        file_name="./run_distributed_accelerate.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                            else:
                                print("worker node")
                                # --rdzv_backend c10d
                                command = (
                                        "accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank {machine_rank} --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                    ).format(
                                        num_processes=world_size*torch.cuda.device_count(),
                                        SLURM_NNODES=world_size,
                                        head_node_ip=master_add,
                                        port=master_port,
                                        machine_rank=rank,
                                        file_name="./run_distributed_accelerate.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                        else:
                            if torch.cuda.device_count() > 1: # multi gpu
                                            #                                     --rdzv_backend c10d \
                                            # --main_process_ip {head_node_ip} \
                                            # --main_process_port {port} \
                                            # --mixed_precision 'fp16' \
                                            # --num_machines {SLURM_NNODES} \
                                            # --num_processes {num_processes}
                                command = (
                                        "accelerate launch --multi_gpu --num_machines {SLURM_NNODES} --machine_rank 0 --num_processes {num_processes} {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                    ).format(
                                        num_processes=world_size*torch.cuda.device_count(),
                                        SLURM_NNODES=world_size,
                                        # head_node_ip=os.environ.get("head_node_ip", master_add),
                                        port=master_port,
                                        file_name="./run_distributed_accelerate.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                                # mixed_precision: fp16
                            elif torch.cuda.device_count() == 1: # one gpu
                                # num_processes = world_size*get_device_count()
                                #    --rdzv_backend c10d \
                                #     --main_process_ip {head_node_ip} \
                                #     --main_process_port {port} \
                                # --num_cpu_threads_per_process=2 --num_processes {num_processes} \
                                #     --num_machines {SLURM_NNODES} \
                                command = (
                                    "accelerate launch {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                ).format(
                                    # num_processes=os.environ.get("num_processes", num_processes),
                                    # SLURM_NNODES=os.environ.get("SLURM_NNODES", world_size),
                                    # head_node_ip=os.environ.get("head_node_ip", master_add),
                                    # port=os.environ.get("port", master_port),
                                    file_name="./run_distributed_accelerate.py",
                                    json_file=json_file,
                                    dataset_path=dataset_path,
                                    channel_log=channel_log,
                                    hf_model_id=hf_model_id,
                                    push_to_hub=push_to_hub,
                                    push_to_hub_token=push_to_hub_token,
                                    action=action
                                )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                                # python3.10 -m pip install -r requirements.txt
                            else: # no gpu
                                command = (
                                        "accelerate launch --cpu {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                    ).format(
                                        file_name="./run_distributed_accelerate.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                print(command)
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                    elif framework == "vllm":
                        if int(world_size) > 1:
                            if int(rank) == 0:
                                print("master node")
                                #  --dynamo_backend 'no' \
                                # --rdzv_backend c10d
                                command = (
                                        "accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank 0 --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                    ).format(
                                        num_processes=world_size*torch.cuda.device_count(),
                                        SLURM_NNODES=world_size,
                                        head_node_ip=master_add,
                                        port=master_port,
                                        file_name="./run_distributed_accelerate.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token
                                    )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                            else:
                                print("worker node")
                                # --rdzv_backend c10d
                                command = (
                                        "accelerate launch --num_processes {num_processes} --num_machines {SLURM_NNODES} --machine_rank {machine_rank} --main_process_ip {head_node_ip} --main_process_port {port} {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                    ).format(
                                        num_processes=world_size*torch.cuda.device_count(),
                                        SLURM_NNODES=world_size,
                                        head_node_ip=master_add,
                                        port=master_port,
                                        machine_rank=rank,
                                        file_name="./run_distributed_accelerate.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                        else:
                            if torch.cuda.device_count() > 1: # multi gpu
                                            #                                     --rdzv_backend c10d \
                                            # --main_process_ip {head_node_ip} \
                                            # --main_process_port {port} \
                                            # --mixed_precision 'fp16' \
                                            # --num_machines {SLURM_NNODES} \
                                            # --num_processes {num_processes}
                                command = (
                                        "accelerate launch --multi_gpu --num_machines {SLURM_NNODES} --machine_rank 0 --num_processes {num_processes} {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                    ).format(
                                        num_processes=world_size*torch.cuda.device_count(),
                                        SLURM_NNODES=world_size,
                                        # head_node_ip=os.environ.get("head_node_ip", master_add),
                                        port=master_port,
                                        file_name="./run_distributed_accelerate.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                                # mixed_precision: fp16
                            elif torch.cuda.device_count() == 1: # one gpu
                                # num_processes = world_size*get_device_count()
                                #    --rdzv_backend c10d \
                                #     --main_process_ip {head_node_ip} \
                                #     --main_process_port {port} \
                                # --num_cpu_threads_per_process=2 --num_processes {num_processes} \
                                #     --num_machines {SLURM_NNODES} \
                                command = (
                                    "accelerate launch {file_name} --training_args_json {json_file}  --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                ).format(
                                    # num_processes=os.environ.get("num_processes", num_processes),
                                    # SLURM_NNODES=os.environ.get("SLURM_NNODES", world_size),
                                    # head_node_ip=os.environ.get("head_node_ip", master_add),
                                    # port=os.environ.get("port", master_port),
                                    file_name="./run_distributed_accelerate.py",
                                    json_file=json_file,
                                    dataset_path=dataset_path,
                                    channel_log=channel_log,
                                    hf_model_id=hf_model_id,
                                    push_to_hub=push_to_hub,
                                    push_to_hub_token=push_to_hub_token,
                                    action=action
                                )
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)
                                # python3.10 -m pip install -r requirements.txt
                            else: # no gpu
                                command = (
                                        "accelerate launch --cpu {file_name} --training_args_json {json_file} --dataset_local {dataset_path} --channel_log {channel_log} --hf_model_id {hf_model_id} --push_to_hub {push_to_hub} --push_to_hub_token {push_to_hub_token} --action {action}"
                                    ).format(
                                        file_name="./run_distributed_accelerate.py",
                                        json_file=json_file,
                                        dataset_path=dataset_path,
                                        channel_log=channel_log,
                                        hf_model_id=hf_model_id,
                                        push_to_hub=push_to_hub,
                                        push_to_hub_token=push_to_hub_token,
                                        action=action
                                    )
                                print(command)
                                process = subprocess.run(
                                    command,
                                    shell=True,
                                    # capture_output=True, text=True).stdout.strip("\n")
                                )
                                #print(process)
                                # run_train(command)

                    output_dir = "./data/checkpoint"
                    print(push_to_hub)
                    import datetime
                    now = datetime.datetime.now()
                    date_str = now.strftime("%Y%m%d")
                    time_str = now.strftime("%H%M%S")
                    version = f'{date_str}-{time_str}'
                    upload_checkpoint(project, version, output_dir)

                train_thread = threading.Thread(target=func_train_model, args=(clone_dir, project_id, token, checkpoint_version, checkpoint_id, dataset_version, dataset_id,model_id,world_size,rank,master_add,master_port,prompt, absolute_path, channel_log, hf_model_id, push_to_hub, push_to_hub_token,host_name))
                train_thread.start()

                return {"message": "train completed successfully"}
            # except Exception as e:
            #     return {"message": f"train failed: {e}"}
        # elif command.lower() == "stop":
        #     subprocess.run(["pkill", "-9", "-f", "llama_recipes/finetuning.py"])
        #     return {"message": "train stop successfully", "result": "Done"}
        elif command.lower() == "tensorboard":
            def run_tensorboard():
                # train_dir = os.path.join(os.getcwd(), "{project_id}")
                # log_dir = os.path.join(os.getcwd(), "logs")
                p = subprocess.Popen(f"tensorboard --logdir /app/data/checkpoint/runs --host 0.0.0.0 --port=6006", stdout=subprocess.PIPE, stderr=None, shell=True)
                out = p.communicate()
                print(out)
            import threading
            tensorboard_thread = threading.Thread(target=run_tensorboard)
            tensorboard_thread.start()
            return {"message": "tensorboardx started successfully"}
        elif command.lower() == "predict":
            try:
                import torch
            # try:
                # checkpoint = kwargs.get("checkpoint")
                # imagebase64 = kwargs.get("image","")
                prompt = kwargs.get("prompt", None)
                model_id = kwargs.get("model_id", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
                text = kwargs.get("text", None)
                token_length = kwargs.get("token_lenght", 30)
                task = kwargs.get("task", "")
                voice = kwargs.get("voice", "")
                max_new_token = kwargs.get("max_new_token", 256)
                temperature = kwargs.get("temperature", 0.9)
                top_k = kwargs.get("top_k", 0.0)
                top_p = kwargs.get("top_p", 0.6)
                world_size = kwargs.get("world_size", 1)
                rank = kwargs.get("rank", 0)
                master_add = kwargs.get("master_add","0.0.0.0")
                master_port = kwargs.get("master_port", "23456")
                framework = kwargs.get("framework", "pytorch")
                hf_token = kwargs.get("hf_token", "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")
                action = "predict"

                import os
                os.environ['HF_TOKEN'] =  hf_token
                if len(voice)>0:
                    import base64
                    import requests
                    import torchaudio

                    def decode_base64_to_audio(base64_audio, output_file="output.wav"):
                        # Giải mã Base64 thành nhị phân
                        audio_data = base64.b64decode(base64_audio)

                        # Ghi dữ liệu nhị phân vào file âm thanh
                        with open(output_file, "wb") as audio_file:
                            audio_file.write(audio_data)
                        return output_file

                    audio_file = decode_base64_to_audio(voice["data"])
                    file_path = "unity_on_device.ptl"

                    if not os.path.exists(file_path):
                        url = "https://huggingface.co/facebook/seamless-m4t-unity-small/resolve/main/unity_on_device.ptl"
                        response = requests.get(url)

                        # Lưu file
                        with open("unity_on_device.ptl", "wb") as f:
                            f.write(response.content)

                    audio_input, _ = torchaudio.load(audio_file) # Load waveform using torchaudio

                    s2st_model = torch.jit.load(file_path)

                    with torch.no_grad():
                        prompt, units, waveform = s2st_model(audio_input, tgt_lang="eng")

                # if not checkpoint:
                if framework == "huggingface":
                    print("accelerate")
                    if int(world_size) > 1:
                        if int(rank) == 0:
                                command = (
                                            "accelerate launch --num_processes {tensor_parallel_size} --num_machines {nnodes} --machine_rank {node_rank} --main_process_ip {host} --main_process_port {port} {file_name} --predict_args={predict_args} --action={action}"
                                        ).format(
                                            tensor_parallel_size=world_size*torch.cuda.device_count(),
                                            hf_model_id=model_id,
                                            port=master_port,
                                            host=master_add,
                                            node_rank=rank,
                                            nnodes=world_size,
                                            predict_args = 'predict_args.json',
                                            file_name="run_distributed_accelerate.py",
                                            action=action
                                        )
                                subprocess.run(
                                            command,
                                            shell=True
                                        )

                        else:
                            print("worker node")

                            command = (
                                "accelerate launch --num_processes {tensor_parallel_size} --num_machines {nnodes} --machine_rank {node_rank} --main_process_ip {host} --main_process_port {port} {file_name} --predict_args={predict_args} --action={action}"
                                ).format(
                                    tensor_parallel_size=world_size*torch.cuda.device_count(),
                                    hf_model_id=model_id,
                                    port=master_port,
                                    host=master_add,
                                    node_rank=rank,
                                    nnodes=world_size,
                                    predict_args = 'predict_args.json',
                                    file_name="run_distributed_accelerate.py",
                                    action=action
                                )
                            subprocess.run(
                                command,
                                shell=True,
                            )
                        import openai

                        client = openai.Client(base_url=f"https://{master_add}:{master_port}/v1", api_key="None")

                        response = client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {"role": "user", "content": text},
                            ],
                            temperature=0,
                            max_tokens=64,
                        )
                        print(response)
                        return {"message": "predict completed successfully", "result": response}
                    else:
                                if torch.cuda.device_count() > 1: # multi gpu

                                    command = (
                                        "accelerate launch --multi_gpu  --num_processes {tensor_parallel_size} --num_machines {nnodes} --machine_rank {node_rank} --main_process_ip {host} --main_process_port {port} {file_name} --predict_args={predict_args} --action={action}"
                                        ).format(
                                            tensor_parallel_size=world_size*torch.cuda.device_count(),
                                            hf_model_id=model_id,
                                            port=master_port,
                                            host=master_add,
                                            node_rank=rank,
                                            nnodes=world_size,
                                            predict_args = 'predict_args.json',
                                            file_name="./run_distributed_accelerate.py",
                                            action=action
                                        )
                                    subprocess.run(
                                        command,
                                        shell=True,
                                    )

                                elif torch.cuda.device_count() == 1: # one gpu
                                    command = (
                                        "accelerate launch --num_processes {tensor_parallel_size} --num_machines {nnodes} --machine_rank {node_rank} --main_process_ip {host} --main_process_port {port} {file_name} --predict_args={predict_args} --action={action}"
                                        ).format(
                                            tensor_parallel_size=world_size*torch.cuda.device_count(),
                                            hf_model_id=model_id,
                                            port=master_port,
                                            host=master_add,
                                            node_rank=rank,
                                            nnodes=world_size,
                                            predict_args = 'predict_args.json',
                                            file_name="./run_distributed_accelerate.py",
                                            action=action
                                        )
                                    subprocess.run(
                                        command,
                                        shell=True,
                                    )

                                else: # no gpu
                                    command = (
                                        "accelerate launch --cpu {file_name} --predict_args={predict_args} --action={action}"
                                        ).format(
                                            tensor_parallel_size=world_size*torch.cuda.device_count(),
                                            hf_model_id=model_id,
                                            port=master_port,
                                            host=master_add,
                                            node_rank=rank,
                                            nnodes=world_size,
                                            predict_args = 'predict_args.json',
                                            file_name="./run_distributed_accelerate.py",
                                            action=action
                                        )
                                    subprocess.run(
                                        command,
                                        shell=True,
                                    )
                elif framework == "sglang":
                    print("sglang")
                    # !python3 -m sglang.launch_server --model tiiuae/Falcon3-1B-Instruct --trust-remote-code --tp 1 --port 30000 --host 0.0.0.0
                    # command = (
                    #             "python3.10 -m sglang.launch_server --model {hf_model_id} --trust-remote-code --tp 1 --port {port} --host {host}"
                    #             ).format(
                    #                 hf_model_id=hf_model_id,
                    #                 port=master_port,
                    #                 host=master_add,
                    #             )
                    # subprocess.run(
                    #     command,
                    #     shell=True)
                    if int(world_size) > 1:
                        if int(rank) == 0:
                            command = (
                                "python3.10 -m sglang.launch_server --model {hf_model_id} --trust-remote-code --tp {tensor_parallel_size} --nnodes {nnodes} --node-rank {node_rank} --dist-init-addr {host}:123456 --port {port} --host 0.0.0.0"
                            ).format(
                               tensor_parallel_size= torch.cuda.device_count(),
                                hf_model_id=model_id,
                                port=master_port,
                                host=master_add,
                                node_rank=rank,
                                nnodes=world_size,
                            )
                            subprocess.run(
                                command,
                                shell=True,
                            )

                        else:
                            print("worker node")

                            command = (
                                    "python3.10 -m sglang.launch_server --model {hf_model_id} --trust-remote-code --tp {tensor_parallel_size} --dist-init-addr {host}:123456 --port {port} --host 0.0.0.0"
                                ).format(
                                    tensor_parallel_size= torch.cuda.device_count(),
                                    hf_model_id=model_id,
                                    port=master_port,
                                    host=master_add,
                                    node_rank=rank,
                                    nnodes=world_size,
                                )
                            subprocess.run(
                                command,
                                shell=True,
                            )
                        # import openai

                        # client = openai.Client(base_url=f"https://{master_add}:{master_port}/v1", api_key="None")

                        # response = client.chat.completions.create(
                        #     model=model_id,
                        #     messages=[
                        #         {"role": "user", "content": text},
                        #     ],
                        #     temperature=0,
                        #     max_tokens=64,
                        # )
                        # print(response)
                        # return {"message": "predict completed successfully", "result": response}
                    else:
                        if torch.cuda.device_count() > 1: # multi gpu

                            command = (
                                "python3.10 -m sglang.launch_server --model {hf_model_id} --trust-remote-code --tp {tensor_parallel_size} --nnodes {nnodes} --node-rank {node_rank} --dist-init-addr {host}:123456 --port {port} --host 0.0.0.0"
                            ).format(
                                tensor_parallel_size= torch.cuda.device_count(),
                                hf_model_id=model_id,
                                port=master_port,
                                host=master_add,
                                node_rank=rank,
                                nnodes=world_size,
                            )
                            subprocess.run(
                                command,
                                shell=True,
                            )

                        elif torch.cuda.device_count() == 1: # one gpu
                            command = (
                                "python3.10 -m sglang.launch_server --model {hf_model_id} --trust-remote-code --tp {tensor_parallel_size} --nnodes {nnodes} --node-rank {node_rank} --dist-init-addr {host}:123456 --port {port} --host 0.0.0.0"
                            ).format(
                                tensor_parallel_size= torch.cuda.device_count(),
                                hf_model_id=model_id,
                                port=master_port,
                                host=master_add,
                                node_rank=rank,
                                nnodes=world_size,
                            )
                            subprocess.run(
                                command,
                                shell=True,
                            )

                        else: # no gpu
                            command = (
                                "python3.10 -m sglang.launch_server --model {hf_model_id} --trust-remote-code --nnodes {nnodes} --node-rank {node_rank} --dist-init-addr {host}:123456 --port {port} --host 0.0.0.0"
                            ).format(
                                tensor_parallel_size=1, #world_size*torch.cuda.device_count(),
                                hf_model_id=model_id,
                                port=master_port,
                                host=master_add,
                                node_rank=rank,
                                nnodes=world_size,
                            )
                            subprocess.run(
                                command,
                                shell=True,
                            )
                     # https://docs.sglang.ai/start/send_request.html
                        # import requests
                        # url = "http://{master_add}:{master_port}/v1/chat/completions"

                        # data = {
                        #     "model":hf_model_id,
                        #     "messages": [{"role": "user", "content": "What is the capital of France?"}],
                        # }

                        # response = requests.post(url, json=data)
                        # print(response.json())

                        # import openai

                        # client = openai.Client(base_url=f"https://{master_add}:{master_port}/v1", api_key="None")

                        # response = client.chat.completions.create(
                        #     model=model_id,
                        #     messages=[
                        #         {"role": "user", "content": text},
                        #     ],
                        #     temperature=0,
                        #     max_tokens=64,
                        # )
                        # print(response)
                        # return {"message": "predict completed successfully", "result": response}
                elif framework == "vllm":
                    print("vllm")
                    # command = (
                    #             "vllm serve {hf_model_id} --tensor-parallel-size {tensor_parallel_size} --enforce-eager --trust-remote-code --port {port} --host {host}"
                    #             ).format(
                    #                 hf_model_id=hf_model_id,
                    #                 tensor_parallel_size=1,
                    #                 port=master_port,
                    #                 host=master_add,
                    #             )
                    # subprocess.run(
                    #     command,
                    #     shell=True)
                    # https://docs.vllm.ai/en/latest/getting_started/quickstart.html
                    # python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 12345 --max-model-len 65536 --trust-remote-code --tensor-parallel-size 8 --quantization moe_wna16 --gpu-memory-utilization 0.97 --kv-cache-dtype fp8_e5m2 --calculate-kv-scales --served-model-name deepseek-reasoner --pipeline-parallel-size 2 --model cognitivecomputations/DeepSeek-R1-AWQ
                    #======== init ddp netwoprk =================
                    if int(world_size) > 1:
                        if int(rank) == 0:
                            command = (
                                "vllm serve {hf_model_id} --tensor-parallel-size {tensor_parallel_size} --pipeline-parallel-size {world_size} --enforce-eager --trust-remote-code --port {port} --host {host}"
                            ).format(
                                tensor_parallel_size=torch.cuda.device_count(),
                                world_size=world_size,
                                hf_model_id=model_id,
                                port=master_port,
                                host=master_add,

                            )
                            subprocess.run(
                                command,
                                shell=True,
                            )

                        else:
                            print("worker node")

                            command = (
                                    "vllm serve {hf_model_id} --tensor-parallel-size {tensor_parallel_size} --pipeline-parallel-size {world_size} --enforce-eager --trust-remote-code --port {port} --host {host}"
                                ).format(
                                    tensor_parallel_size=torch.cuda.device_count(),
                                    world_size=world_size,
                                    hf_model_id=model_id,
                                    port=master_port,
                                    host=master_add,
                                )
                            subprocess.run(
                                command,
                                shell=True,
                            )
                        # from openai import OpenAI
                        # # Set OpenAI's API key and API base to use vLLM's API server.
                        # openai_api_key = "EMPTY"
                        # openai_api_base = f"https://{master_add}:{master_port}/v1"

                        # client = OpenAI(
                        #     api_key=openai_api_key,
                        #     base_url=openai_api_base,
                        # )

                        # chat_response = client.chat.completions.create(
                        #     model=model_id,
                        #     messages=[
                        #         {"role": "system", "content": "You are a helpful assistant."},
                        #         {"role": "user", "content": text},
                        #     ]
                        # )
                        # print("Chat response:", chat_response)
                        # return {"message": "predict completed successfully", "result": chat_response}
                    else:
                        if torch.cuda.device_count() > 1: # multi gpu

                            command = (
                                "vllm serve {hf_model_id} --tensor-parallel-size {tensor_parallel_size} --pipeline-parallel-size {world_size} --enforce-eager --trust-remote-code --port {port} --host {host}"
                            ).format(
                                tensor_parallel_size=torch.cuda.device_count(),
                                    world_size=world_size,
                                hf_model_id=model_id,
                                port=master_port,
                                host=master_add,
                            )
                            subprocess.run(
                                command,
                                shell=True,
                            )
                            # from openai import OpenAI
                            # # Set OpenAI's API key and API base to use vLLM's API server.
                            # openai_api_key = "EMPTY"
                            # openai_api_base = f"https://{master_add}:{master_port}/v1"

                            # client = OpenAI(
                            #     api_key=openai_api_key,
                            #     base_url=openai_api_base,
                            # )

                            # chat_response = client.chat.completions.create(
                            #     model=model_id,
                            #     messages=[
                            #         {"role": "system", "content": "You are a helpful assistant."},
                            #         {"role": "user", "content": text},
                            #     ]
                            # )
                            # print("Chat response:", chat_response)
                            # return {"message": "predict completed successfully", "result": chat_response}
                        elif torch.cuda.device_count() == 1: # one gpu
                            command = (
                               "vllm serve {hf_model_id} --tensor-parallel-size {tensor_parallel_size} --pipeline-parallel-size {world_size} --enforce-eager --trust-remote-code --port {port} --host {host}"
                            ).format(
                                tensor_parallel_size=torch.cuda.device_count(),
                                    world_size=world_size,
                                hf_model_id=model_id,
                                port=master_port,
                                host=master_add,
                            )
                            subprocess.run(
                                command,
                                shell=True,
                            )

                        else: # no gpu
                            command = (
                               "vllm serve {hf_model_id} --enforce-eager --trust-remote-code --gpu-memory-utilization 0.97 --port {port} --host {host}"
                            ).format(
                                tensor_parallel_size=world_size*torch.cuda.device_count(),
                                hf_model_id=model_id,
                                port=master_port,
                                host=master_add,
                            )
                            subprocess.run(
                                command,
                                shell=True,
                            )
                        #======== end init ddp netwoprk =================
                        #call request predict
                        # curl http://{master_add}:{master_port}/v1/chat/completions \
                        # -H "Content-Type: application/json" \
                        # -d '{
                        #     "model":hf_model_id,
                        #     "messages": [
                        #         {"role": "system", "content": "You are a helpful assistant."},
                        #         {"role": "user", "content": "Who won the world series in 2020?"}
                        #     ]
                        # }'
                        # from openai import OpenAI
                        # # Set OpenAI's API key and API base to use vLLM's API server.
                        # openai_api_key = "EMPTY"
                        # openai_api_base = f"https://{master_add}:{master_port}/v1"

                        # client = OpenAI(
                        #     api_key=openai_api_key,
                        #     base_url=openai_api_base,
                        # )

                        # chat_response = client.chat.completions.create(
                        #     model=model_id,
                        #     messages=[
                        #         {"role": "system", "content": "You are a helpful assistant."},
                        #         {"role": "user", "content": text},
                        #     ]
                        # )
                        # print("Chat response:", chat_response)
                        # return {"message": "predict completed successfully", "result": chat_response}
                elif framework == "horovod":
                    print("horovod")
                    import torch
                    from torch.optim import AdamW
                    import transformers
                    from transformers import AutoConfig, AutoModelForSequenceClassification #, get_linear_schedule_with_warmup
                    from torch.utils.data import DataLoader, SequentialSampler
                    from torch.utils.data.distributed import DistributedSampler

                    from petastorm.spark import SparkDatasetConverter, make_spark_converter
                    from petastorm import TransformSpec

                    import horovod.torch as hvd
                    from sparkdl.horovod import log_to_driver
                    from sparkdl import HorovodRunner
                    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, './cache')
                    test_data_folder = './test_data'
                    sdf_tokens_test = spark.read.parquet(os.path.join(test_data_folder, 'tokens_test_data'))
                    converter_test = make_spark_converter(sdf_tokens_test, dtype=None)
                    def model_inference():

                        # initiate Horovod in the worker node
                        ######
                        hvd.init()
                        ######

                        # ranks are the unique Horovod process identifiers
                        # local_rank refers to the process(es) running on a node
                        # rank refers to the processes running on the entire cluster (all participating nodes)
                        # world_size is the size of the cluster in number of nodes in a cluster
                        ######
                        local_rank = hvd.local_rank()
                        rank = hvd.rank()
                        world_size = hvd.size()
                        ######

                        # rank 0 usually is the node that triggers the MPI job execution in a Horovod cluster
                        # when we need to run something only in a single node in the cluster, we usually use the rank 0 node
                        ######
                        inference_folder = './inference_outputs'
                        if rank == 0:
                            os.makedirs(inference_folder, exist_ok = True)
                            log_to_driver(f'Cluster size: {world_size} GPUs')
                        ######

                        # set the GPU device for PyTorch
                        ######
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        if device.type == 'cuda':
                            torch.cuda.set_device(local_rank)

                        torch.cuda.empty_cache()
                        ######

                        # set random seeds for reproducibility
                        ######
                        seed_val = 111
                        random.seed(seed_val)
                        np.random.seed(seed_val)
                        torch.manual_seed(seed_val)
                        torch.cuda.manual_seed_all(seed_val)
                        ######

                        # here we instantiate the pre-trained model from Hugging Face, but load it with the fine-tuned model weights
                        ######
                        model_type = 'microsoft/deberta-v3-base'
                        max_length = 128
                        hidden_dropout_prob = 0.
                        attention_probs_dropout_prob = 0.
                        num_labels = 2
                        batch_size = 32

                        model_folder = './model_outputs'
                        config = AutoConfig.from_pretrained(model_type, hidden_dropout_prob=hidden_dropout_prob,
                                                            attention_probs_dropout_prob=attention_probs_dropout_prob, num_labels=num_labels)
                        model = AutoModelForSequenceClassification.from_pretrained(model_folder, config=config).to(device)
                        ######

                        # initiate the lists to be used to store the predicted values
                        ######
                        ids = []
                        labels = []
                        predicted_labels = []
                        probs = []
                        ######

                        # we create the PyTorch DataLoader object from the Petastorm SparkDatasetConverter object we defined earlier
                        with converter_test.make_torch_dataloader(cur_shard=rank, shard_count=world_size, batch_size=batch_size, num_epochs=1) as test_dataloader:

                            # here is the loop where we iterate over the testing data and use the fine-tuned model to get predictions
                            ######
                            start_time = time.time()

                            for _, batch in enumerate(test_dataloader):
                                input_ids = batch['input_ids'].to(device)
                                token_type_ids = batch['token_type_ids'].to(device)
                                attention_mask = batch['attention_mask'].to(device)
                                id = batch['id']
                                label = batch['label'].to(device)

                                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=label)
                                prob = torch.softmax(outputs.logits, dim=1).tolist()
                                predicted_label = list(map(lambda x: 0 if x[1] < 0.5 else 1, prob))

                                ids += id.tolist()
                                labels += label.tolist()
                                probs += prob
                                predicted_labels += predicted_label

                            end_time = time.time()
                            log_to_driver('\nWorker %i: Duration: %f' % (rank, round(end_time-start_time, 3)))
                            ######

                            # we create a pandas dataframe with the ids and labels from the testing data, and the corresponding predicted labels and probabilities
                            # then we aggregate the dataframes of all worker nodes in a list and save it as a pickle file
                            # we also save the average model scoring duration across all worker nodes
                            ######
                            data_pred = {'id': ids, 'label': labels, 'predicted_label': predicted_labels, 'prob': probs}
                            df_pred = pd.DataFrame(data_pred)
                            df_pred = hvd.allgather_object(df_pred, name='df_pred')
                            duration = hvd.allreduce(torch.tensor(round(end_time-start_time, 3))).detach().numpy()

                            if rank == 0:
                                with open(inference_folder + '/df_pred.pkl', 'wb') as f:
                                    pickle.dump(df_pred, f)
                                with open(inference_folder + '/duration.pkl', 'wb') as f:
                                    pickle.dump(duration, f)
                            ######
                    #======== init ddp netwoprk =================
                    num_workers = world_size
                    hr = HorovodRunner(np=num_workers,hosts=f"{master_add}:{torch.cuda.device_count()}",use_gloo='gloo',use_mpi= 'mpi')
                    hr.run(model_inference)


                    #======== end init ddp netwoprk =================
                else:
                    print("pytorch")
                    #======== init ddp netwoprk =================
                    if int(world_size) > 1:
                        if int(rank) == 0:
                            print("CUDA is available. world_size > 1 and rank=0")
                            command = (
                                        "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                        "--master_addr {master_addr} --master_port {master_port} {file_name} --backend {backend} --rank {node_rank} --channel_log {channel_log} --hf_model_id {hf_model_id} --hf_hub_token {hf_hub_token}"
                                    ).format(
                                        nnodes=int(world_size),
                                        node_rank= int(rank),
                                        nproc_per_node=torch.cuda.device_count(),
                                        master_addr=master_add,
                                        master_port=master_port,
                                        file_name="./run_distributed_pytorch.py",
                                        predict_args = 'predict_args.json',
                                        backend = "nccl",
                                        channel_log="",
                                        hf_model_id=model_id,
                                        hf_hub_token=hf_token #'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'
                                    )
                            print(command)
                            response = subprocess.run(
                                command,
                                shell=True
                            )
                            if response.returncode == 0:
                                response_info_lines = response.stdout.strip()
                            else:
                                response_info_lines = ""
                            return {"message": "predict completed successfully", "result": response_info_lines}

                        else:
                            print("worker node")

                            print("CUDA is available. > 1")
                            command = (
                                        "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                        "--master_addr {master_addr} --master_port {master_port} {file_name} --backend {backend} --predict_args {predict_args} --rank {node_rank} --channel_log {channel_log} --hf_model_id {hf_model_id} --hf_hub_token {hf_hub_token}"
                                    ).format(
                                         nnodes=int(world_size),
                                        node_rank= int(rank),
                                        nproc_per_node=torch.cuda.device_count(),
                                        master_addr=master_add,
                                        master_port=master_port,
                                        file_name="./run_distributed_pytorch.py",
                                        predict_args = 'predict_args.json',
                                        backend = "nccl",
                                        channel_log="",
                                        hf_model_id=model_id,
                                        hf_hub_token=hf_token #'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'
                                    )
                            response = subprocess.run(
                                command,
                                shell=True
                            )
                            if response.returncode == 0:
                                response_info_lines = response.stdout.strip()
                            else:
                                response_info_lines = ""
                            return {"message": "predict completed successfully", "result": response_info_lines}

                    else:
                        if torch.cuda.device_count() > 1: # multi gpu
                            print("CUDA is available. > 1")
                            command = (
                                        "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                                        "--master_addr {master_addr} --master_port {master_port} {file_name} --backend {backend} --predict_args {predict_args} --rank {node_rank} --channel_log {channel_log} --hf_model_id {hf_model_id} --hf_hub_token {hf_hub_token}"
                                    ).format(
                                         nnodes=int(world_size),
                                        node_rank= int(rank),
                                        nproc_per_node=torch.cuda.device_count(),
                                        master_addr=master_add,
                                        master_port=master_port,
                                        file_name="./run_distributed_pytorch.py",
                                        predict_args = 'predict_args.json',
                                        backend = "nccl",
                                        channel_log="",
                                        hf_model_id=model_id,
                                        hf_hub_token=hf_token #'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'
                                    )
                            print(command)
                            response = subprocess.run(
                                command,
                                shell=True
                            )
                            if response.returncode == 0:
                                response_info_lines = response.stdout.strip()
                            else:
                                response_info_lines = ""
                            return {"message": "predict completed successfully", "result": response_info_lines}
                        elif torch.cuda.device_count() == 1: # one gpu

                            print("CUDA is available.")

                            _model = pipeline(
                                task,
                                model=model_id,
                                device_map="auto",  # Hoặc có thể thử "cpu" nếu không ổn,
                                max_new_tokens=int(max_new_token),
                                temperature=float(temperature),
                                top_k=float(top_k),
                                top_p=float(top_p),
                                token=hf_token #'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'
                            )

                            predictions = []

                            if task == "question-answering":
                                print("Question answering")
                                if text and prompt:
                                    generated_text = qa_with_context(_model, text, prompt)
                                elif text and not prompt:
                                    generated_text = qa_without_context(_model, text)
                                else:
                                    generated_text = qa_with_context(_model, prompt)

                            elif task == "text-classification":
                                generated_text = text_classification(_model, text, prompt)

                            elif task == "summarization":
                                generated_text = text_summarization(_model, text)

                            else:
                                if not prompt or prompt == "":
                                    prompt = text

                                result = _model(prompt, max_length=token_length)
                                generated_text = result[0]['generated_text']

                            predictions.append({
                                'result': [{
                                    'from_name': "generated_text",
                                    'to_name': "text_output",
                                    'type': 'textarea',
                                    'value': {
                                        'text': [generated_text]
                                    }
                                }],
                                'model_version': ""
                            })

                            return {"message": "predict completed successfully", "result": predictions}

                        else: # no gpu

                            print("No GPU available, using CPU.")
                            _model = pipeline(
                                task,
                                model=model_id,
                                device_map="cpu",
                                max_new_tokens=int(max_new_token),
                                temperature=float(temperature),
                                top_k=float(top_k),
                                top_p=float(top_p),
                                token=hf_token #'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'
                            )
                            predictions = []

                            if task == "question-answering":
                                print("Question answering")
                                if text and prompt:
                                    generated_text = qa_with_context(_model, text, prompt)
                                elif text and not prompt:
                                    generated_text = qa_without_context(_model, text)
                                else:
                                    generated_text = qa_with_context(_model, prompt)

                            elif task == "text-classification":
                                generated_text = text_classification(_model, text, prompt)

                            elif task == "summarization":
                                generated_text = text_summarization(_model, text)

                            else:
                                if not prompt or prompt == "":
                                    prompt = text

                                result = _model(prompt, max_length=token_length)
                                generated_text = result[0]['generated_text']

                            predictions.append({
                                'result': [{
                                    'from_name': "generated_text",
                                    'to_name': "text_output",
                                    'type': 'textarea',
                                    'value': {
                                        'text': [generated_text]
                                    }
                                }],
                                'model_version': ""
                            })

                            return {"message": "predict completed successfully", "result": predictions}
                        #======== end init ddp netwoprk =================



            except Exception as e:
                return {"message": f"predict failed:{e}", "result": None}
        elif command.lower() == "prompt_sample":
                task = kwargs.get("task", "")
                if task == "question-answering":
                    prompt_text = f"""
                   Here is the context: 
                    {{context}}

                    Based on the above context, provide an answer to the following question: 
                    {{question}}

                    Answer:
                    """
                elif task == "text-classification":
                   prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """

                elif task == "summarization":
                    prompt_text = f"""
                    Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

                    Text: 
                    {{context}}

                    Summary:
                    """
                return {"message": "prompt_sample completed successfully", "result":prompt_text}
        elif command.lower() == "stop":
            subprocess.run(["pkill", "-9", "-f", "accelerate launch"])
            return {"message": "Done", "result": "Done"}
        elif command.lower() == "action-example":
            return {"message": "Done", "result": "Done"}
        else:
            return {"message": "command not supported", "result": None}


            # return {"message": "train completed successfully"}

    def model(self,  **kwargs):

        import gradio as gr
        from transformers import pipeline
        task = kwargs.get("task", "text-generation")
        model_id = kwargs.get("model_id", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
        project_id = project

        print(f'''\
        Project ID: {project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}''')
        from huggingface_hub import login
        hf_access_token = kwargs.get("hf_access_token", "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")
        login(token = hf_access_token)
        def load_model(task,model_id, project_id, temperature=None, top_p=None, top_k=None, max_new_token=None):
            from huggingface_hub import login
            hf_access_token = kwargs.get("hf_access_token", "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")
            login(token = hf_access_token)
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float16

                print("CUDA is available.")

                if not temperature:
                    _model = pipeline(
                    task,
                    model=model_id, #"meta-llama/Llama-4-Scout-17B-16E-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
                    torch_dtype=dtype,
                    device_map="auto",  # Hoặc có thể thử "cpu" nếu không ổn,
                    # max_new_tokens=256,
                    token = "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"
                    )
                else:
                    _model = pipeline(
                        task,
                        model=model_id, #"meta-llama/Llama-4-Scout-17B-16E-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
                        torch_dtype=dtype,
                        device_map="auto",  # Hoặc có thể thử "cpu" nếu không ổn,
                        # max_new_tokens=256,
                        token = "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",
                        max_new_tokens=int(max_new_token),
                        temperature=float(temperature),
                        top_k=float(top_k),
                        top_p=float(top_p)
                    )
            else:
                print("No GPU available, using CPU.")
                if not temperature:
                    _model = pipeline(
                    task,
                    model=model_id, #"meta-llama/Llama-4-Scout-17B-16E-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
                    torch_dtype=dtype,
                    device_map="auto",  # Hoặc có thể thử "cpu" nếu không ổn,
                    # max_new_tokens=256,
                    token = "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"
                    )
                else:
                    _model = pipeline(
                        task,
                        model=model_id, #"meta-llama/Llama-4-Scout-17B-16E-Instruct", #"meta-llama/Llama-3.2-3B", meta-llama/Llama-3.3-70B-Instruct
                        device_map="cpu",
                        # max_new_tokens=256,
                        token = "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",
                        max_new_tokens=int(max_new_token),
                        temperature=float(temperature),
                        top_k=float(top_k),
                        top_p=float(top_p),
                    )
            # try:
            #     channel_deploy = f'project/{project_id}/deploy-history'
            #     client, sub = setup_client(channel_deploy)
            #     send_message(sub,{"refresh": True})
            #     client.disconnect()  # Đóng kết nối client
            # except Exception as e:
            #     print(e)

            return _model

        # load_model(task,model_id,project_id)

        def generate_response(input_text, temperature, top_p, top_k, max_new_token):
            load_model("text-generation", model_id, project_id, temperature, top_p, top_k, max_new_token)
            # _model = load_model(model_id)
            # prompt_text = f"""
            #        Here is the context: 
            #         {{context}}

            #         Based on the above context, provide an answer to the following question using only a single word or phrase from the context without repeating the question or adding any extra explanation: 
            #         {{question}}

            #         Answer:
            #         """
            # prompt = prompt_text


            # if not prompt_text or prompt_text == "":
            #     prompt = input_text
            messages = [
                {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                {"role": "user", "content": input_text},
            ]
            # outputs = _model(
            #     messages,
            #     max_new_tokens=256,
            # )
            result = _model(messages, max_length=1024)
            generated_text = result[0]['generated_text']

            # if input_text and prompt_text:
            #     generated_text = qa_with_context(_model, input_text, prompt_text)
            # elif input_text and not prompt_text:
            #     generated_text = qa_without_context(_model, prompt_text)
            # else:
            #     generated_text = qa_with_context(_model, prompt_text)

            return generated_text

        def summarization_response(input_text, temperature, top_p, top_k, max_new_token):
            load_model("text-generation", model_id, project_id, temperature, top_p, top_k, max_new_token)
            # _model = load_model(model_id)
            generated_text = text_summarization(_model, input_text)

            return generated_text

        def text_classification_response(input_text,categories_text, temperature, top_p, top_k, max_new_token):
            load_model("text-generation", model_id, project_id, temperature, top_p, top_k, max_new_token)
            # prompt_text = f"""
            #         Summarize the following text into a single, concise paragraph focusing on the key ideas and important points:

            #         Text: 
            #         {{context}}

            #         Summary:
            #         """

            # _model = load_model(model_id)
            generated_text = text_classification(_model, input_text, categories_text)
            return generated_text

        def question_answering_response(context_textbox,question_textbox, temperature, top_p, top_k, max_new_token):
            load_model("text-generation", model_id, project_id, temperature, top_p, top_k, max_new_token)
            # _model = load_model(model_id)
            if input_text and question_textbox:
                generated_text = qa_with_context(_model, context_textbox, question_textbox)
            elif context_textbox and not question_textbox:
                generated_text = qa_without_context(_model, question_textbox)
            else:
                generated_text = qa_with_context(_model, question_textbox)

            return generated_text

        def chatbot_continuous_chat(history, user_input, temperature, top_p, top_k, max_new_token):
            load_model("text-generation", model_id, project_id, temperature, top_p, top_k, max_new_token)
            generated_text = chatbot_with_history(_model, history, user_input)
            return generated_text

        with gr.Blocks() as demo_text_generation:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        # prompt_text = gr.Textbox(label="Prompt text")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")
            # gr.on(
            #     triggers=[input_text.submit, temperature.submit, top_p.submit, top_k.submit, max_new_token.submit, btn.click],
            #     fn=generate_response,
            #     inputs=[input_text, temperature, top_p, top_k, max_new_token],
            #     outputs=output_text,
            #     api_name=task,
            # )
            btn.click(
                fn=generate_response,
                inputs=[input_text, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )

        with gr.Blocks() as demo_summarization:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                        # prompt_text = gr.Textbox(label="Prompt text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")
            # gr.on(
            #     triggers=[input_text.submit, temperature.submit, top_p.submit, top_k.submit, max_new_token.submit, btn.click],
            #     fn=summarization_response,
            #     inputs=[input_text, temperature, top_p, top_k, max_new_token],
            #     outputs=output_text,
            #     api_name=task,
            # )
            btn.click(
                fn=summarization_response,
                inputs=[input_text, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )

        with gr.Blocks() as demo_question_answering:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        context_textbox = gr.Textbox(label="Context text")
                        question_textbox = gr.Textbox(label="Question text")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )

                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text =   gr.Textbox(label="Response:")


            # gr.Examples(
            #    inputs=[input_text],
            #     outputs=output_text,
            #     fn=question_answering_response,
            #     api_name=False,
            # )

            # gr.on(
            #     triggers=[context_textbox.submit,question_textbox.submit, temperature.submit, top_p.submit, top_k.submit, max_new_token.submit, btn.click],
            #     fn=question_answering_response,
            #     inputs=[context_textbox, question_textbox, temperature, top_p, top_k, max_new_token],
            #     outputs=output_text,
            #     api_name=task,
            # )
            btn.click(
                fn=question_answering_response,
                inputs=[context_textbox, question_textbox, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )

        with gr.Blocks() as demo_text_classification:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        categories_text = gr.Textbox(label="Categories text")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                # with gr.Column():
                #     with gr.Group():
                #         input_text = gr.Textbox(label="Input text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Response:")


            # gr.on(
            #     triggers=[input_text.submit,categories_text.submit, temperature.submit, top_p.submit, top_k.submit, max_new_token.submit, btn.click],
            #     fn=text_classification_response,
            #     inputs=[input_text, categories_text, temperature, top_p, top_k, max_new_token],
            #     outputs=output_text,
            #     api_name=task,
            # )
            btn.click(
                fn=text_classification_response,
                inputs=[input_text, categories_text, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )

        def sentiment_classifier(text, temperature, top_p, top_k, max_new_token):
            try:
                sentiment_classifier = pipeline("sentiment-analysis", temperature=temperature, top_p=top_p, top_k=top_k, max_new_token=max_new_token)
                sentiment_response = sentiment_classifier(text)
                # label = sentiment_response[0]['label']
                # score = sentiment_response[0]['score']
                print(sentiment_response)
                import json
                return json.dumps(sentiment_response)
            except Exception as e:
                return str(e)

        with gr.Blocks() as demo_sentiment_analysis:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                    btn = gr.Button("Submit")
                with gr.Column():

                    label_text = gr.Label(label="Label: ")
                    score_text = gr.Label(label="Score: ")

            # gr.on(
            #     triggers=[input_text.submit, temperature.submit, top_p.submit, top_k.submit, max_new_token.submit, btn.click],
            #     fn=sentiment_classifier,
            #     inputs=[input_text, temperature, top_p, top_k, max_new_token],
            #     outputs=[label_text, score_text],
            #     api_name=task,
            # )
            btn.click(
                fn=sentiment_classifier,
                inputs=[input_text, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )

        def predict_entities(input_text, categories_text, temperature, top_p, top_k, max_new_token):
            #  # Initialize the text-generation pipeline with your model
            # pipe = pipeline(task, model=model_id)
            # # Use the loaded model to identify entities in the text
            # entities = pipe(text)
            # # Highlight identified entities in the input text
            # highlighted_text = text
            # for entity in entities:
            #     entity_text = text[entity['start']:entity['end']]
            #     replacement = f"<span style='border: 2px solid green;'>{entity_text}</span>"
            #     highlighted_text = highlighted_text.replace(entity_text, replacement)
            # return highlighted_text
            load_model("text-generation", model_id, project_id, temperature, top_p, top_k, max_new_token)
            generated_text = text_ner(_model, input_text, categories_text)
            return generated_text

        with gr.Blocks() as demo_ner:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text")
                        categories_text = gr.Textbox(label="Categories text")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                    # with gr.Group():
                    #     input_text = gr.Textbox(label="Input text")
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Response:")

            # gr.Examples(
            #     inputs=[input_text],
            #     outputs=output_text,
            #     fn=generate_response,
            #     api_name=False,
            # )

            # gr.on(
            #     triggers=[input_text.submit,categories_text.submit, temperature.submit, top_p.submit, top_k.submit, max_new_token.submit, btn.click],
            #     fn=predict_entities,
            #     inputs=[input_text, categories_text, temperature, top_p, top_k, max_new_token],
            #     outputs=output_text,
            #     api_name=task,
            # )
            btn.click(
                fn=predict_entities,
                inputs=[input_text, categories_text, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )

        with gr.Blocks() as demo_text2text_generation:
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        input_text = gr.Textbox(label="Input text", placeholder="Enter your text here")
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                        top_p = gr.Slider(
                            label="Top_p",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.6
                        )
                        top_k = gr.Slider(
                            label="Top_k",
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=0
                        )
                        max_new_token = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=256
                        )
                    btn = gr.Button("Submit")
                with gr.Column():
                    output_text = gr.Textbox(label="Output text")

            # Gắn sự kiện với nút Submit
            btn.click(
                fn=generate_response,
                inputs=[input_text, temperature, top_p, top_k, max_new_token],
                outputs=output_text
            )

        with gr.Blocks() as demo_chatbot:
            chatbot = gr.Chatbot(type="messages")
            with gr.Row():
                user_input = gr.Textbox(label="Your message", placeholder="Type your message here...")
            with gr.Row():
                temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=100.0,
                            step=0.1,
                            value=0.9
                        )
                top_p = gr.Slider(
                    label="Top_p",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.6
                )
                top_k = gr.Slider(
                    label="Top_k",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=0
                )
                max_new_token = gr.Slider(
                    label="Max new tokens",
                    minimum=1,
                    maximum=1024,
                    step=1,
                    value=256
                )
            with gr.Row():
                btn = gr.Button("Send")

            # Bind function to button click
            btn.click(
                fn=chatbot_continuous_chat,
                inputs=[chatbot, user_input, temperature, top_p, top_k, max_new_token],  # Thêm các tham số vào hàm
                outputs=[chatbot, user_input]  # Cập nhật lại chatbot với phản hồi mới và xóa trường nhập liệu
            )


        DESCRIPTION = """\
        # LLM UI
        This is a demo of LLM UI.
        """
        with gr.Blocks(css="style.css") as demo:
            gr.Markdown(DESCRIPTION)

            with gr.Tabs():
                if task == "text-generation":
                    with gr.Tab(label=task):
                        demo_text_generation.render()
                elif task == "summarization":
                    with gr.Tab(label=task):
                        demo_summarization.render()
                elif task == "question-answering":
                    with gr.Tab(label=task):
                        demo_question_answering.render()
                elif task == "text-classification":
                    with gr.Tab(label=task):
                            demo_text_classification.render()
                elif task == "sentiment-analysis":
                    with gr.Tab(label=task):
                        demo_sentiment_analysis.render()
                elif task == "ner":
                    with gr.Tab(label=task):
                        demo_ner.render()
                # elif task == "fill-mask":
                #   with gr.Tab(label=task):
                #         demo_fill_mask.render()
                elif task == "text2text-generation":
                    with gr.Tab(label=task):
                        demo_text2text_generation.render()
                elif task == "chat":
                    with gr.Tab(label=task):
                        demo_chatbot.render()
                else:
                    return {"share_url": "", 'local_url': ""}

        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)

        return {"share_url": share_url, 'local_url': local_url}

    def model_trial(self, project, **kwargs):
        import gradio as gr

        return {"message": "Done", "result": "Done"}


        css = """
        .feedback .tab-nav {
            justify-content: center;
        }

        .feedback button.selected{
            background-color:rgb(115,0,254); !important;
            color: #ffff !important;
        }

        .feedback button{
            font-size: 16px !important;
            color: black !important;
            border-radius: 12px !important;
            display: block !important;
            margin-right: 17px !important;
            border: 1px solid var(--border-color-primary);
        }

        .feedback div {
            border: none !important;
            justify-content: center;
            margin-bottom: 5px;
        }

        .feedback .panel{
            background: none !important;
        }


        .feedback .unpadded_box{
            border-style: groove !important;
            width: 500px;
            height: 345px;
            margin: auto;
        }

        .feedback .secondary{
            background: rgb(225,0,170);
            color: #ffff !important;
        }

        .feedback .primary{
            background: rgb(115,0,254);
            color: #ffff !important;
        }

        .upload_image button{
            border: 1px var(--border-color-primary) !important;
        }
        .upload_image {
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }
        .upload_image .wrap{
            align-items: center !important;
            justify-content: center !important;
            border-style: dashed !important;
            width: 500px;
            height: 345px;
            padding: 10px 10px 10px 10px
        }

        .webcam_style .wrap{
            border: none !important;
            align-items: center !important;
            justify-content: center !important;
            height: 345px;
        }

        .webcam_style .feedback button{
            border: none !important;
            height: 345px;
        }

        .webcam_style .unpadded_box {
            all: unset !important;
        }

        .btn-custom {
            background: rgb(0,0,0) !important;
            color: #ffff !important;
            width: 200px;
        }

        .title1 {
            margin-right: 90px !important;
        }

        .title1 block{
            margin-right: 90px !important;
        }

        """

        with gr.Blocks(css=css) as demo:
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown(
                        """
                        # Theme preview: `AIxBlock`
                        """
                    )

            import numpy as np
            def predict(input_img):
                import cv2
                result = self.action(project, "predict",collection="",data={"img":input_img})
                print(result)
                if result['result']:
                    boxes = result['result']['boxes']
                    names = result['result']['names']
                    labels = result['result']['labels']

                    for box, label in zip(boxes, labels):
                        box = [int(i) for i in box]
                        label = int(label)
                        input_img = cv2.rectangle(input_img, box, color=(255, 0, 0), thickness=2)
                        # input_img = cv2.(input_img, names[label], (box[0], box[1]), color=(255, 0, 0), size=1)
                        input_img = cv2.putText(input_img, names[label], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                return input_img

            def download_btn(evt: gr.SelectData):
                print(f"Downloading {dataset_choosen}")
                return f'<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"><a href="/my_ml_backend/datasets/{evt.value}" style="font-size:50px"> <i class="fa fa-download"></i> Download this dataset</a>'

            def trial_training(dataset_choosen):
                print(f"Training with {dataset_choosen}")
                result = self.action(project, "train",collection="",data=dataset_choosen)
                return result['message']

            def get_checkpoint_list(project):
                print("GETTING CHECKPOINT LIST")
                print(f"Proejct: {project}")
                import os
                checkpoint_list = [i for i in os.listdir("my_ml_backend/models") if i.endswith(".pt")]
                checkpoint_list = [f"<a href='./my_ml_backend/checkpoints/{i}' download>{i}</a>" for i in checkpoint_list]
                if os.path.exists(f"my_ml_backend/{project}"):
                    for folder in os.listdir(f"my_ml_backend/{project}"):
                        if "train" in folder:
                            project_checkpoint_list = [i for i in os.listdir(f"my_ml_backend/{project}/{folder}/weights") if i.endswith(".pt")]
                            project_checkpoint_list = [f"<a href='./my_ml_backend/{project}/{folder}/weights/{i}' download>{folder}-{i}</a>" for i in project_checkpoint_list]
                            checkpoint_list.extend(project_checkpoint_list)

                return "<br>".join(checkpoint_list)

            def tab_changed(tab):
                if tab == "Download":
                    get_checkpoint_list(project=project)

            def upload_file(file):
                return "File uploaded!"

            with gr.Tabs(elem_classes=["feedback"]) as parent_tabs:
                with gr.TabItem("Image", id=0):
                    with gr.Row():
                        gr.Markdown("## Input", elem_classes=["title1"])
                        gr.Markdown("## Output", elem_classes=["title1"])

                    gr.Interface(predict, gr.Image(elem_classes=["upload_image"], sources="upload", container = False, height = 345,show_label = False),
                                gr.Image(elem_classes=["upload_image"],container = False, height = 345,show_label = False), allow_flagging = False
                    )


                # with gr.TabItem("Webcam", id=1):
                #     gr.Image(elem_classes=["webcam_style"], sources="webcam", container = False, show_label = False, height = 450)

                # with gr.TabItem("Video", id=2):    
                #     gr.Image(elem_classes=["upload_image"], sources="clipboard", height = 345,container = False, show_label = False)

                # with gr.TabItem("About", id=3):  
                #     gr.Label("About Page")

                with gr.TabItem("Trial Train", id=2):
                    gr.Markdown("# Trial Train")
                    with gr.Column():
                        with gr.Column():
                            gr.Markdown("## Dataset template to prepare your own and initiate training")
                            with gr.Row():
                                #get all filename in datasets folder
                                if not os.path.exists(f"./datasets"):
                                    os.makedirs(f"./datasets")
                                datasets = [(f"dataset{i}", name) for i, name in enumerate(os.listdir('./datasets'))]

                                dataset_choosen = gr.Dropdown(datasets, label="Choose dataset", show_label=False, interactive=True, type="value")
                                # gr.Button("Download this dataset", variant="primary").click(download_btn, dataset_choosen, gr.HTML())
                                download_link = gr.HTML("""
                                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
                                        <a href='' style="font-size:24px"><i class="fa fa-download" ></i> Download this dataset</a>""")

                                dataset_choosen.select(download_btn, None, download_link)

                                #when the button is clicked, download the dataset from dropdown
                                # download_btn
                            gr.Markdown("## Upload your sample dataset to have a trial training")
                            # gr.File(file_types=['tar','zip'])
                            gr.Interface(predict, gr.File(elem_classes=["upload_image"],file_types=['tar','zip']),
                                gr.Label(elem_classes=["upload_image"],container = False), allow_flagging = False
                    )
                            with gr.Row():
                                gr.Markdown(f"## You can attemp up to {2} FLOps")
                                gr.Button("Trial Train", variant="primary").click(trial_training, dataset_choosen, None)

                # with gr.TabItem("Download"):
                #     with gr.Column():
                #         gr.Markdown("## Download")
                #         with gr.Column():
                #             gr.HTML(get_checkpoint_list(project))

        gradio_app, local_url, share_url = demo.launch(share=True, quiet=True, prevent_thread_lock=True, server_name='0.0.0.0',show_error=True)

        return {"share_url": share_url, 'local_url': local_url}

    def download(self, project, **kwargs):
        from flask import send_from_directory,request
        file_path = request.args.get('path')
        print(request.args)
        return send_from_directory(os.getcwd(), file_path)
    # def download(self, project, **kwargs):
    #     from flask import send_file
    #     file_path = kwargs.get("path","")
    #     if os.path.exists(os.path.join("/app/", file_path)):
    #         return send_file(os.path.join("/app/", file_path), as_attachment=True)
    #     else:
    #         print("file dont exist")
    #         return {"msg": "file dont exist"}