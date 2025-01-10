import time
import torch
import numpy as np

from transformers import LlamaTokenizer, LlamaForCausalLM

from llama import LLaMAModelConfig, LLaMA
from quant import QuantConfig, QuantType

def main(model_path="llama-13b-hf"):
    assert LlamaTokenizer, "tinyllama tokenizer load failed, pip install transformer"

    quant_config = QuantConfig(type=QuantType.AutoInt8)
    quant_config = None
    load_model_pt = True
    origin_model_dir = f"/home/tianxin/data/LLM/{model_path}"

    model_config_13b: LLaMAModelConfig = {
        "num_layers": 40,
        "dim_model": 5120,
        "num_heads": 40,
        "dim_head": 128,
        "dim_ff": 13824,
        "vocab_size": 32000,
    }

    if model_path == "llama-13b-hf":
        model = LLaMA(
            f"{origin_model_dir}.ckpt",
            origin_model_dir,
            -1,
            memory_limit=30 << 30,
            model_config = model_config_13b,
            quant_config = quant_config,
            load_model = not load_model_pt,
            weight_transposed =not load_model_pt,
        )


    print("done!!!")

if __name__ == "__main__":
    main("llama-13b-hf")

