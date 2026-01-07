# Benchmarking


## First use vllm to serve the local model


```sh
local_llm_path="/path/to/model"
local_llm_name="local_llm"
port=8081
tool_call_parser="hermes" # for qwen3, we use hermes

vllm serve $local_llm_path \
  --tensor-parallel-size 1 \
  --port $port \
  --enable-auto-tool-choice \
  --tool-call-parser $tool_call_parser \
  --served-model-name $local_llm_name
```

## If you are using lora adapters

```sh

base_model_path=/path/to/qwen3-14b
lora_adapter_path=/path/to/qwen3-14b-ft-adapter
lora_name=qwen3-14b-ft-with-thinking

vllm serve $base_model_path \
  --tensor-parallel-size 1 \
  --port $port \
  --enable-auto-tool-choice \
  --tool-call-parser $tool_call_parser \
  --gpu-memory-utilization 0.7 \
  --enable-lora \
  --lora-modules $lora_name=$lora_adapter_path
```

## Install tau2-bench

```sh
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench
pip install .
export TAU2_DATA_DIR=$PWD/data
```

## Run benchmark


```sh
benchmark_type="retail"
agent_provider="hosted_vllm"
user_provider="together_ai"
user_agent_name="moonshotai/Kimi-K2-Instruct-0905"
lora_name="qwen3-14b-ft-with-thinking"
port=8082
temperature=0.6

cd tau2-bench

export TAU2_DATA_DIR=$PWD/data
export TOGETHER_API_KEY=your_together_api_key

tau2 run \
--domain $benchmark_type \
--agent-llm $agent_provider/$lora_name \
--agent-llm-args "{\"temperature\": $temperature, \"api_base\": \"http://localhost:$port/v1\"}" \
--user-llm $user_provider/$user_agent_name \
--num-trials 1 \
--max-concurrency 5 \
--max-errors 10000
```

This will produce a lot of non-critical errors saying there is no billing information for these models. It can be ignored.

Sometimes, the test has to be done in 2 chunks. For the first chunk I would add `--task-ids $(echo {0..50})` and for the second chunk I would do `--task-ids $(echo {51..113})`. This is the number of tasks for the retail domain.