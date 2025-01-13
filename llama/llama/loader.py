import concurrent.futures
import os
import re
import time
import json
import glob
import torch
from abc import ABC, abstractmethod

class ModelLoader(ABC):
    @abstractmethod
    def fetch_parameter(self, name):
        ...

    @staticmethod
    def convert_quant_dict(state_dict):
        if 'quant_state' not in state_dict:
            return state_dict
        state = state_dict['state']
        quant_state = state_dict['quant_state']
        state_dict = {}
        for name, param in state.items():
            if name in quant_state:
                prefix = name.rsplit('.', 1)[0]
                state_dict[prefix + ".qunat_weight"] = param
                state_dict[prefix + ".scale_weight"] = quant_state[name]['scales']
                state_dict[prefix + ".zero_weight"] = quant_state[name]['qzeros']
            elif isinstance(param, torch.Tensor):
                state_dict[name] = param
        return state_dict

    @staticmethod
    def load_safetensors(model_dir, pattern="*.safetensors", parallel=None):
        if not hasattr(torch, "float8_e4m3fn"):
            torch.float8_e4m3fn = torch.int8
        from safetensors.torch import load_file
        files = sorted(glob.glob(f"{model_dir}/{pattern}"))
        if not files:
            raise ValueError(f"No safetensors files found in: {model_dir}")
        state_dict = {}
        if parallel is None:
            parallel = model_dir.startswith("/tmp") and os.environ.get("DISABLE_PARALLEL_LOAD", "0") != "1"
        if parallel:
            print(f"########## parallel load_clone {len(files)} files ##########")
            t0 = time.time()
            def load_clone(f):
                d1 = load_file(f)
                return {k : torch.clone(v) for k, v in d1.items()}
            with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
                futures = [executor.submit(load_clone, f) for f in files]
                for f in futures:
                    state_dict.update(f.result())
            print(f"########## Done load_clone {len(files)} files in {time.time() - t0:.1f} seconds ##########")
            return state_dict

        for f in files:
            state_dict.update(load_file(f))
        return state_dict

    @staticmethod
    def load_pt(model_dir):
        state_dict = {}
        if os.path.isfile(f"{model_dir}"):
            state_dict = torch.load(f"{model_dir}", map_location="cpu")
        elif os.path.isfile(f"{model_dir}/pytorch_model.pt"):
            state_dict = torch.load(f"{model_dir}/pytorch_model.pt", map_location="cpu")
        else:
            pt_files = sorted(glob.glob(f"{model_dir}/pytorch_model*.bin"))
            if not pt_files:
                pt_files = sorted(glob.glob(f"{model_dir}/caterpillar_*.pt"))
            if not pt_files:
                pt_files = sorted(glob.glob(f"{model_dir}/cpm_*pt"))
            if not pt_files and glob.glob(f"{model_dir}/*.safetensors"):
                return ModelLoader.load_safetensors(model_dir)
            if not pt_files:
                raise ValueError(f"No checkpoint found in: {model_dir}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(torch.load, f, "cpu") for f in pt_files]
                for f in futures:
                    state_dict.update(f.result())
        return state_dict

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

    @staticmethod
    def load_safetensors(model_dir, pattern="*.safetensors"):
        state_dict = ModelLoader.load_safetensors(model_dir, pattern)
        return LLaMALoader.convert_quant_dict(state_dict)

    @staticmethod
    def load_pt(model_dir):
        state_dict = ModelLoader.load_pt(model_dir)
        return LLaMALoader.convert_quant_dict(state_dict)

    @staticmethod
    def _replace_name(s):
        s = re.sub("model.embed_tokens.weight", "token_embedding.weight", s)
        s = re.sub("model.nor.weight", "output_layernorm.weight", s)
        s = re.sub(
            "model.layers.([0-9]+).input_layernorm.(weight|scales|qweight|qzeros)",
            "layers.\\1.ln_attn.\\2",
            s,
        )

        return "llama." + s

