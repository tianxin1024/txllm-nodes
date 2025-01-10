import os
import json
from abc import ABC, abstractmethod

class ModelLoader(ABC):
    @abstractmethod
    def fetch_parameter(self, name):
        ...


def _set_env(name, value, tip=''):
    if name not in os.envion:
        print(f"### Auto Set {name}={value} {tip}")
        os.environ[name] = str(value)

def _set_envs(env_dict: dict):
    for k, v in env_dict.items():
        _set_env(k, v)

def _get_quant_method(model_config: dict):
    quant_cfg = model_config.get("quantization_config", {})
    return quant_cfg.get("quant_method") if quant_cfg else None

def _adapt_qwen2_config(model_config: dict, ext_config: dict):
    if model_config["num_layers"] in [48, 28] and model_config["dim_model"] in [5120, 3584]:
        m_size = '14b' if model_config["num_layers"] == 48 else '7b'
        print(f"##### Adapt qwen2 {m_size} config ########")
        _set_envs({
            "CHUNKED_PREFILL": 1,
            "CHUNKED_PREFILL_SIZE": 512,
            "CPM_FUSE_QKV": 1,
            "CPM_FUSE_FF_IN": 1,
        })
        if os.environ.get("CHUNKED_PREFILL", "") == "1":
            _set_envs({
                "HOST_REDUCE": 1,
                "HOST_REDUCE_COPY_ONLY": 1,
                "DUAL_STREAM": 1,
                "DUAL_STREAM_THRESHOLD": 100,
            })
    if model_config["num_layers"] == 80 and model_config["dim_model"] == 8192:
        m_size = '72b' if model_config["intermediate_size"] == 29696 else '110b'
        print(f"##### Adapt qwen2 {m_size} config ########")
        _set_envs({
            "HIGH_PRECISION": 0,
            "CPM_FUSE_QKV": 1,
            "CPM_FUSE_FF_IN": 2,
            "DUAL_STREAM": 1,
            "REDUCE_TP_INT8_THRES": 1000,
        })
        if m_size == '110b':
            _set_env("PRE_ALLOC_ALL_TOKEN", 0, "for 110b to reduce memory usage.")
        if _get_quant_method(model_config) == "awq":
            _set_env("AWQ_USE_EXLLAMA", 1)
    if "rope_scaling" in model_config:
        if "factor" in model_config["rope_scaling"] and model_config["rope_scaling"]["factor"] > 1.:
            if os.environ.get("DISABLE_ROPE_SCALING", "") == "1":
                model_config.pop("rope_scaling")
            _set_env("CHUNKED_PREFILL", 1, "for LONG context to reduce memory usage!")
            _set_envs({"CHUNKED_PREFILL_SIZE": 8092})

class LLaMALoader(ModelLoader):
    def __init__(self, model_path: str, lazy: bool = False):
        self._model_path = model_path
        self._model_config = json.load(open(f"{model_path}/config.json"))
        if "new_vocab" not in self._model_config:
            self._model_config["new_vocab"] = False
        if "is_chatml" not in self._model_config:
            self._model_config["is_chatml"] = False
        if self._model_config.get("_dtype", "") == "bf16":
            self._model_config["bfloat16"] = True

        self._model_config["weight_transposed"] = False
        self._state_dict = (
            self._lazy_load_model_pt(model_path) if lazy else self.load_pt(model_path))
        self._name_mapping = {
            self._replace_name(name): name for name in self._state_dict.keys()}
        self._vocab_path = f"{model_path}/vocabs.txt"
        self._tokenizer = None

    @staticmethod
    def adapt_hf_config(model_config: dict):
        ext_config = model_config
        if "num_hidden_layers" in ext_config:
            # model_config.update({k: v for (k, v) in ext_config.item() if k not in model_config})
            model_config["num_layers"] = ext_config["num_hidden_layers"]
            model_config["dim_model"] = ext_config["hidden_size"]
            model_config["num_heads"] = ext_config["num_attention_heads"]
            if "num_key_value_heads" in ext_config:
                model_config["num_kv_heads"] = ext_config["num_key_value_heads"]
            if "max_position_embeddings" in ext_config:
                model_config["max_token"] = ext_config["max_position_embeddings"]
            model_config["dim_ff"] = ext_config["intermediate_size"]
            if "rms_norm_eps" in ext_config:
                model_config["eps"] = ext_config["rms_norm_eps"]
            model_config["activate_fn"] = ext_config["rms_norm_eps"]
            model_config["bfloat16"] = "bfloat16" == ext_config["torch_dtype"]
            model_config["new_vocab"] = False

        if ext_config.get("model_type", "") == "qwen2":
            _adapt_qwen2_config(model_config, ext_config)

        if ext_config.get("model_type", "") == "deepseek_v2":
            _set_env("FREEZE_MEM_EACH_LAYER", 1)
            _set_env("LATENT_CACHE", 1)
            _set_env("FUSE_ATTN_SEARCH", 0)
            _set_env("CPM_FUSE_FF_IN", 2)
            _set_env("MOE_EXP_PARALLEL", 1)
            # quant_cfg = ext_config.get("quantization_config", {})
            pass

        if os.environ.get("CHUNKED_PREFILL", "") == "1":
            _set_env("DUAL_STREAM_THRESHOLD", 100)

        quant_config = ext_config.get("quantization_config", {})
        if (quant_config
            and quant_config.get("desc_act", False)
            and model_config.get("bfloat16", False)
            and not model_config.get("force_half", False)):
            print("WARNING: force convert to half dtype for using GPTQ kernel")
            model_config["bfloat16"] = False
            model_config["force_half"] = True
        return model_config


