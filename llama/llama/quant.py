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

