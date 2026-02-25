
<h1 align="center" style="font-size: 30px;"><strong><em>Long Context, Less Focus:</em></strong> A Scaling Gap in LLMs Revealed through Privacy and Personalization</h1>
<p align="center">
    <a href="https://arxiv.org/pdf/2602.15028">Paper</a>
    ·
    <a href="https://github.com/SafeRL-Lab/PAPerBench">Code</a>
   ·
    <a href="https://huggingface.co/datasets/PAPer-project/PAPerBench">Dataset</a>
    ·    
    <a href="https://github.com/SafeRL-Lab/PAPerBench/issues">Issue</a>
  </p>
</div>




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

## Citation

If you use PAPerBench in your research, please cite:

```bibtex
@article{gu2026long,
  title={Long Context, Less Focus: A Scaling Gap in LLMs Revealed through Privacy and Personalization},
  author={Gu, Shangding},
  journal={arXiv preprint arXiv:2602.15028},
  year={2026}
}
```
