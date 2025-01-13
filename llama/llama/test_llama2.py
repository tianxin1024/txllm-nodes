import time
import torch
import numpy as np

from transformers import LlamaTokenizer, LlamaForCausalLM

from llama import LLaMAModelConfig, LLaMA
from quant import QuantConfig, QuantType



def load_model(ver="7b", load_model_pt=False):
    t0 = time.time()
    assert LlamaTokenizer, "llama tokenizer load failed, pip install transformer"
    model_dir = f"/home/tianxin/data/LLM/llama2"

    model_config_7b: LLaMAModelConfig = {
        "num_layers": 32,
        "dim_model": 4096,
        "num_heads": 32,
        "dim_head": 128,
        "dim_ff": 11008,
        "vocab_size": 32000,
        "eps": 1e-5,
    }

    if ver.startswith("7b"):
        model = LLaMA(f"{model_dir}/model.ckpt",
                      model_dir,
                      -1,
                      memory_limit = 30 << 30,
                      model_config = model_config_7b,)
    else:
        raise ValueError(f"Unknown version {ver}")

    if load_model_pt:
        model.load_model_pt(f"{model_dir}")

    print(f"model load finished in {time.time() - t0} seconds")
    return model

def main(ver="7b"):
    model = load_model(ver, False)


    print("done!!!")

if __name__ == "__main__":
    main("7b-chat")

