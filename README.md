# The Effect of Context Length on Privacy and Personalization: Revealing a Scaling Gap

[Dataset](https://huggingface.co/datasets/Anonymous-199/PAPerBench)


Install packages

```pip install -r requirements.txt```

Run VLLM server
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
