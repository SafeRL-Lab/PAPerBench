# for open source models
export VLLM_BASE_URL="http://localhost:8002/v1"
python eval_personalization_mcq.py \
  --input your_data.jsonl \
  --outdir your_results/ \
  --models "meta-llama/Llama-4-Scout-17B-16E-Instruct" \
  --temperature 0.0 \
  --max_tokens 8

# for closed source models
export USE_OPENAI=0
export LITELLM_BASE_URL="your_litellm_link/v1"
export LITELLM_API_KEY="sk-XXX"
python /home/shangdinggu/nvidia_server/agentic_web/live_rl/opal_benchmark/personal/build_2stage_data/eval_personalization_litllm.py \
  --input your_data.jsonl \
  --outdir your_results/ \
  --models "openai/gpt-5.2" \
  --temperature 0.0 --max_tokens 8 \
