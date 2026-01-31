# for open source models:

export VLLM_API_KEY="EMPTY"   
python eva_privacy.py \
  --provider vllm \
  --base-url http://localhost:8002/v1 \
  --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --mcq-field pii_count_mcq \
  --mcq-field privacy_aggregate_mcq \
  --temperature 0.0 \
  --max-tokens 8 \
  --input your_data.jsonl \
  --output your_results.jsonl

# for closed source models
unset VLLM_API_KEY
unset VLLM_BASE_URL
export OPENAI_API_KEY="sk-XXX"
python eva_privacy.py \
  --provider openai \
  --model gpt-5.2 \
  --mcq-field pii_count_mcq \
  --mcq-field privacy_aggregate_mcq \
  --temperature 0.0 \
  --max-tokens 8 \
  --input your_data.jsonl \
  --output your_results.jsonl
