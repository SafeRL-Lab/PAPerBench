# for open source models
export VLLM_BASE_URL="http://localhost:8002/v1"
python eval_personalization_mcq.py \
  --input your_data.jsonl \
  --outdir your_results/ \
  --models "meta-llama/Llama-4-Scout-17B-16E-Instruct" \
  --temperature 0.0 \
  --max_tokens 8
