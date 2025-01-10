import enum
import os
from typing_extensions import TypedDict


@enum.unique
class QuantType(enum.Enum):
    NoQuant = 0
    AbsMax = 1  # Load from quantized int8 weights and float16 scales
    AutoInt8 = 2  # Load from float16 weights, do int8 quantization during loading model.
    Int4 = 3  # Load from quantized int4 weights and float16 scales and zeros
    AutoInt4 = 4  # Only for speed test
    GPTQ = 5
    AWQ = 6
    FP8 = 7
    GPTQ_Marlin = 8
    AWQ_Marlin = 9


class QuantConfig(TypedDict, total=False):
    type: QuantType

    # We can skip quant project_k and v; which occupy only 1% of multi-group attention model
    quant_weight_kv: int  # -1: auto, 0: no, 1: yes. default: auto
    act_order: bool  # see https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda
    group_size: int
    sym: bool

    @staticmethod
    def adapt_hf_config(config, hf_config):
        cfg = config or QuantConfig()
        hf_config = hf_config.get("quantization_config", "")
        if hf_config:
            quant_method = hf_config.get("quant_method", "")
            if os.environ.get("MARLIN_KERNEL") == "1" and quant_method == "gptq":
                print("Use Marlin kernel for GPTQ")
                cfg["type"] = QuantType.GPTQ_Marlin
                cfg["sym"] = hf_config.get("sym", False)
                os.environ["NEED_DEQUANT_WEIGHT"] = "0"
                if hf_config.get("bits", 4) != 4:
                    raise ValueError("Only bits=4 is supported")
                if hf_config.get("is_marlin_format", False):
                    raise ValueError(f"Unsupported Marlin {quant_method}")
            elif quant_method == "awq":
                # set AWQ_USE_EXLLAMA to force use exllama kernel, which is faster in decoding
                cfg["type"] = QuantType.GPTQ if os.environ.get("AWQ_USE_EXLLAMA") == "1" else QuantType.AWQ
                os.environ["NEED_DEQUANT_WEIGHT"] = "1"
                if hf_config.get("bits", 4) != 4:
                    raise ValueError("Only bits=4 is supported")
            elif quant_method == "gptq":
                cfg["type"] = QuantType.GPTQ
                cfg["sym"] = hf_config.get("sym", False)
                os.environ["NEED_DEQUANT_WEIGHT"] = "1"
                if hf_config.get("bits", 4) != 4:
                    raise ValueError("Only bits=4 is supported")
                if hf_config.get("is_marlin_format", False):
                    # cfg["type"] = QuantType.Marlin
                    raise ValueError(f"Unsupported Marlin {quant_method}")
            elif quant_method == "fp8":
                cfg["type"] = QuantType.FP8
            else:
                raise ValueError(f"Unsupported quant_method {quant_method}")
        # if hf_config.get("desc_act", False):
        #     cfg["act_order"] = True
        if cfg.get("act_order", False):
            os.environ["GPTQ_KERNEL_ALGO"] = "0"  # use exllama kernel
            os.environ["FUSE_GPTQ_MOE"] = "0"
            os.environ["MOE_DYN_SHARED"] = "0"
        if "group_size" in hf_config:
            cfg["group_size"] = hf_config["group_size"]
        return cfg



