import time
import torch
import numpy as np

from transformers import LlamaTokenizer, LlamaForCausalLM

from llama import LLaMAModelConfig, LLaMA
from quant import QuantConfig, QuantType


def load_model(ver="1b", load_model_pt=False):
    t0 = time.time()
    assert LlamaTokenizer, "llama tokenizer load failed, pip install transformer"
    model_dir = f"/home/tianxin/data/LLM/TinyLlama-1.1B-Chat-v1.0/{ver}"

    model_config_tiny: LLaMAModelConfig = {
        "num_layers": 22,      # 模型层数
        "dim_model": 2048,     # 隐藏层维度
        "num_heads": 16,       # 注意力头数
        "dim_head": 128,       # 注意力头维度
        "dim_ff": 5632,        # 前馈网络维度
        "vocab_size": 32000,   # 词汇表大小
        "eps": 1e-5,           # 层归一化epsilon
    }

    if ver.startswith("tinyllama-1b"):
        model = LLaMA(f"{model_dir}/model.safetensors",
                      model_dir,
                      -1,
                      memory_limit = 2 << 30,
                      model_config = model_config_tiny,)
    else:
        raise ValueError(f"Unknown version {ver}")

    if load_model_pt:
        model.load_model_safetensors(f"{model_dir}")

    print(f"model load finished in {time.time() - t0} seconds")
    return model

def main(ver="1b"):
    model = load_model(ver, True)


    print("done!!!")

if __name__ == "__main__":
    main("tinyllama-1b")

