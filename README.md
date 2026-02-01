# The Effect of Context Length on Privacy and Personalization: Revealing a Scaling Gap

## Download dataset from the link: [Dataset](https://huggingface.co/datasets/Anonymous-199/PAPerBench)

## Create an environment (requires Conda installation): 

Use the following command to create a new Conda environment named robustgymnasium with Python 3.10:

```
conda create -n paperbench  python=3.10
```

## Activate the newly created environment:

```
conda activate paperbench
```

## Install dependency packages:


```pip install -r requirements.txt```

## Run VLLM server
```
export CUDA_VISIBLE_DEVICES=4,5,6,7   
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --host 0.0.0.0 \
  --port 8002 \
  --tensor-parallel-size 4 \
  --max-model-len 328816 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

## For personalization evaluation

```
bash paperbench/personalization/eva_personalization.sh
```

## For privacy evaluation
```
bash paperbench/privacy/eva_privacy.sh
```
