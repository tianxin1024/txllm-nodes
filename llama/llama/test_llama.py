import os
import time
import torch
import numpy as np

from transformers import LlamaTokenizer, LlamaForCausalLM

from llama import LLaMAModelConfig, LLaMA
from quant import QuantConfig, QuantType
from dynamic_batch import GeneratorArg, DynamicBatchConfig, DynamicBatchGenerator

os.environ["DYN_BATCH_DEBUG"] = "1"

def main():
    t0 = time.time()
    assert LlamaTokenizer, "tinyllama tokenizer load failed, pip install transformer"
    model_dir = "/home/tianxin/data/LLM/TinyLlama-1.1B-Chat-v1.0/tinyllama-1b"

    model_config_tiny: LLaMAModelConfig = {
        "num_layers": 2,      # 模型层数
        "dim_model": 2048,     # 隐藏层维度
        "num_heads": 32,       # 注意力头数
        "num_kv_heads": 4,
        "dim_head": 64,       # 注意力头维度
        "dim_ff": 5632,        # 前馈网络维度
        "vocab_size": 32000,   # 词汇表大小
        "model_type": 'llama',
        "torch_dtype": 'float16',
        "eps": 1e-5,           # 层归一化epsilon
    }

    model = LLaMA(f"{model_dir}/model.safetensors",
                  model_dir,
                  -1,
                  memory_limit = 2 << 30,
                  model_config = model_config_tiny,)

    model.load_model_safetensors(f"{model_dir}")
    print(f">>>Load model '{model_dir}' finished in {time.time() - t0:.2f} seconds<<<")

    arg = GeneratorArg(beam_size=1, max_length=100, repetition_penalty=1.0)

    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

    batch_config = DynamicBatchConfig(flash_attention=False)
    with DynamicBatchGenerator(batch_config, model) as generator:
        print("....")
        req_result = generator.generate(messages, arg)
        print("...." * 100)
        print(req_result)

    print("done!!!")

if __name__ == "__main__":
    # run tinyllama
    main()

