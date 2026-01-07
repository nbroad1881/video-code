# Finetune for Tool Calling

This video is about how to finetune an open model to improve tool calling. Tool calling gives the model the ability to interact with other APIs and it is an essential feature in agents.  

Finetuning can boost performance noticeably depending on the quality and quantity of the data.


## Contents
  
- [Tau2 Benchmarking Instructions](./benchmark.md)  
  - Starting local vllm server  
  - Running tau2-bench  
- [Creating a training dataset with the original apigen dataset](./prepare-apigen-dataset.ipynb)  
- [Creating a training dataset with the apigen-with-thinking dataset](./prepare-apigen-with-thinking.ipynb)  
  - A modified version of the original apigen dataset with thinking traces  



### Chat App

Here is a basic demo of using the model as a customer service agent in retail. It can provide information about products and orders.


#### Steps

1. Make sure local model is running on vllm

#### Merged model

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

#### Lora adapter

```sh

base_model_path=/path/to/qwen3-14b
lora_adapter_path=/path/to/qwen3-14b-ft-adapter
local_llm_name=qwen3-14b-ft-with-thinking

vllm serve $base_model_path \
  --tensor-parallel-size 1 \
  --port $port \
  --enable-auto-tool-choice \
  --tool-call-parser $tool_call_parser \
  --gpu-memory-utilization 0.7 \
  --enable-lora \
  --lora-modules $local_llm_name=$lora_adapter_path
```

2. Launch streamlit app ([code here](./chat_app.py)) 

```sh
streamlit run chat_app.py -- --vllm-base-url http://localhost:$port/v1 --vllm-model $local_llm_name
```

3. Go to streamlit app in browser: http://localhost:8501

The lefthand side has information about what tools are available, and some sample data points in the mock database. The user can add their own information or pretend to be one of the users in the existing database.

Here are some sample queries:

- I would like to see the status of my order. It is order... (see sidebar for sample order numbers)
- I would like more info about a product. The product number is ... (see sidebar for sample product numbers)
- Please cancel my order
- Tell me how to return my order

