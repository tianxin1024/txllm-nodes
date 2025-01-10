import os
import json

from typing import (Any, Dict, List, Optional, Tuple, Union, overload,
                    Callable, NamedTuple,)
from transformers import AutoTokenizer
from typing_extensions import TypedDict
from config import DistConfig
from quant import QuantConfig
from loader import LLaMALoader
from llm_nodes import ModelConfig
import llm_nodes

print(llm_nodes)
print(dir(llm_nodes))


class LLaMAModelConfig(TypedDict):
    num_layers: int
    dim_model: int
    num_heads: int
    dim_ff: int
    vocab_size: int
    eps: float
    num_kv_keads: int
    new_vocab: bool
    bfloat: bool
    pos_bias_type: str
    activate_fn: str
    scale_weights: bool
    weight_transposed: bool
    max_token: int
    tie_lm_head: bool
    rope_theta: float

def _get_config(model_config) -> LLaMAModelConfig:
    cfg = {
        "num_layers": 32,
        "dim_model": 4096,
        "num_heads": 32,
        "dim_head": 128,
        "dim_ff": 11008,
        "vocab_size": 32000,
        "eps": 1e-6,
        "pos_bias_type": "rotary",
        "activate_fn": "silu",
        "scale_weights": False,
        "weight_transposed": False,
        "max_token": 4096,
        "tie_lm_head": False,
        "rope_theta": 10000.0,
    }
    if model_config is not None:
        model_type = model_config.get("model_type", "cpm_caterpillar")
        if model_type == "cpm_dragonfly":
            cfg["dim_head"] = 64
            cfg["tie_lm_head"] = True
            if "num_experts" in model_config:
                cfg["moe_num_experts"] = model_config["num_experts"]
        cfg.update(model_config)
        if model_type == "cpm_caterpillar" and "scale" in model_config:
            cfg["scale_weights"] = model_config["scale"]
    if cfg.get("_dtype", "") == "bf16":
        cfg["bfloat16"] = True
    if "tie_word_embeddings" in cfg:
        cfg["tie_lm_head"] = cfg.get("tie_word_embeddings", False)
    # cfg["eps"] = 1e-6
    if "quantization_config" not in cfg:
        if not os.environ.get("CPM_FUSE_QKV"):
            os.environ["CPM_FUSE_QKV"] = "1"
        if not os.environ.get("CPM_FUSE_FF_IN"):
            os.environ["CPM_FUSE_FF_IN"] = "1"
    if not os.environ.get("HIGH_PRECISION"):
        # Compute_32F
        os.environ["HIGH_PRECISION"] = "1"
    if not os.environ.get("W4_A8_M_THRES"):
        os.environ["W4_A8_M_THRES"] = "1000"
    print(f"[DEV]Config: "
          f'HIGH_PRECISION={os.environ.get("HIGH_PRECISION")}'
          f'; DUAL_STREAM={os.environ.get("DUAL_STREAM")}'
          f'; CPM_FUSE_QKV={os.environ.get("CPM_FUSE_QKV")}'
          f'; CPM_FUSE_FF_IN={os.environ.get("CPM_FUSE_FF_IN")}'
          f'; REDUCE_TP_INT8_THRES={os.environ.get("REDUCE_TP_INT8_THRES")}'
          f'; W4_INT8_ALGO={os.environ.get("W4_INT8_ALGO")}'
          f'; W4_FP8_ALGO={os.environ.get("W4_FP8_ALGO")}'
          )
    return cfg

class LLaMA:
    def __init__(self, 
                 model_path: str,
                 vocab_path: str,
                 device_id: int = 0,
                 memory_limit: int = 0,
                 model_config: Optional[LLaMAModelConfig] = None,
                 quant_config: Optional[QuantConfig] = None,
                 parallel: Union[DistConfig, int, bool] = 0,
                 tokenizer=None,
                 **kwargs):
        self._config = _get_config(LLaMALoader.adapt_hf_config(model_config))
        self._quant_config = QuantConfig.adapt_hf_config(quant_config, self._config)
        dist_config = DistConfig.adapt(parallel)
        print(f"dist_config: parallel={dist_config.parallel}")

        self._init_tokenizer(vocab_path, tokenizer)
        c_config = llm_nodes.ModelConfig(self._config)


    def _init_tokenizer(self, vocab_path: str, tokenizer):
        self.tokenizer_config = {}
        if tokenizer is not None:
            self._tokenizer = tokenizer
            return

        self._tokenizer = AutoTokenizer.from_pretrained(vocab_path)
        self._tokenizer.bos_token_id = self._config.get("bos_token_id", None) or 1
        self._tokenizer.eos_token_id = self._config.get("eos_token_id", None) or 2
        tokenizer_config_path = f"{vocab_path}/tokenizer_config.json"
        if os.path.exists(tokenizer_config_path):
            self.tokenizer_config = json.load(open(tokenizer_config_path))
