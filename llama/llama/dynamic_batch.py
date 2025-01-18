import os
import json
import logging
from typing import List, Optional, Union, Tuple
from threading import Thread

import llm_nodes


class DynamicBatchConfig:
    def __init__(self, 
                 max_batch=20,
                 max_beam_size=4,
                 task_queue_size=8,
                 max_total_token=4096,
                 seed=0,
                 eos_id=1,
                 bos_id=0,
                 unk_id=0,
                 first_batch=1,
                 sort_by_len=0,
                 nccl=-1,
                 rag_buffer=True,
                 ignore_eos=False,
                 keep_eos=False,
                 reserved_work_mem_mb=int(os.environ.get("reserved_work_mem_mb", 1250)),
                 high_precision=1,
                 flash_attention=True,
                 enable_prompt_caching=False):
        self.max_batch = max_batch
        self.max_beam_size = max_beam_size
        self.task_queue_size = max(task_queue_size, first_batch)
        self.max_total_token = max_total_token
        self.seed = seed
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.unk_id = unk_id
        self.first_batch = first_batch
        self.sort_by_len = int(sort_by_len)
        self.nccl = int(nccl)
        self.rag_buffer = bool(rag_buffer)
        self.ignore_eos = bool(ignore_eos)
        self.keep_eos = keep_eos
        self.reserved_work_mem_mb = reserved_work_mem_mb
        self.high_precision = high_precision
        self.flash_attention = flash_attention
        self.enable_prompt_caching = enable_prompt_caching or "1" == os.environ.get("enable_prompt_caching", "")
        if enable_prompt_caching and not rag_buffer:
            raise ValueError("prompt_caching must use with rag buffer")


class GeneratorArg:
    def __init__(self, 
                 beam_size: int = 1,
                 max_length: int = 100,
                 repetition_penalty: float = 1.05,
                 ngram_penalty: float = 1.0,
                 seed: int = 0,
                 temperature: float = 1.0,
                 num_results: int = 1,
                 top_p: float = 1.0,           # use random sample if top_p < 1. or top_k > 0
                 top_k: int = 0,
                 bee_answer_multi_span: Optional[bool] = None,
                 presence_penalty: float = 0.,
                 top_logprobs: int = 0,):
        self.beam_size = beam_size
        self.max_length = max_length
        self.presence_penalty = presence_penalty
        self.repetition_penalty = 1. if presence_penalty else repetition_penalty
        self.ngram_penalty = 1. if presence_penalty else ngram_penalty
        self.seed = seed
        self.temperature = float(temperature)
        self.num_results = int(num_results)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.bee_answer_multi_span = bee_answer_multi_span
        self.top_logprobs = top_logprobs

        if self.is_random:
            self.seed = seed or 42
        elif not self.is_random and self.seed:
            logging.warning("BeamSearch ignore seed")
            self.seed = 0

    def copy(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    @property
    def is_random(self):
        return self.top_p < 1. or self.top_k > 0

    def with_max_length(self, new_len):
        obj = self.copy()
        obj.max_length = new_len
        return obj

class DynamicBatchGenerator:
    def __init__(self, config: DynamicBatchConfig, model):
        self.config = config
        self.model = model
        model = model._base if hasattr(model, "_base") else model
        self.c_model = model._model
        self._tokenizer = model._tokenizer if hasattr(model, "_tokenizer") else None
        self._is_llama = 'llama' in model.__class__.__name__.lower()
        self._is_bee = 'CPMBee' in model.__class__.__name__
        if self._is_bee:
            self.config.bos_id = 0
            self.config.eos_id = 1
        elif self._is_llama:
            # self.config.eos_id = 2
            self.config.bos_id = self._tokenizer.bos_token_id
            self.config.eos_id = self._tokenizer.eos_token_id
            if hasattr(model, "tokenizer_config"):
                tokenizer_cfg: dict = model.tokenizer_config
                if "chat_template" in tokenizer_cfg and "eos_token" in tokenizer_cfg and \
                        tokenizer_cfg["eos_token"] != "</s>" and \
                        "added_tokens_decoder" in tokenizer_cfg:
                    chat_eos_token = tokenizer_cfg["eos_token"]
                    for token_id, added_tokens in tokenizer_cfg["added_tokens_decoder"].items():
                        if chat_eos_token == added_tokens["content"]:
                            self.config.eos_id = int(token_id)
                            print(f"Set chat model eos_id to {token_id}")

        model_config = model._config if hasattr(model, "_config") and isinstance(model._config, dict) else {}
        max_token = model_config.get("max_token", 4096)

        self._c_generator = llm_nodes.BatchGenerator(self.config.c_config(), self.c_model)

        self._thread = None
        self._do_verify = int(os.environ.get("VERIFY_MAX_TOKEN", 1)) > 0

    def start(self):
        def run_wrapper():
            try:
                self._c_generator.run()
            except Exception as e:
                if self._do_verify:
                    print("Verify max_token failed! please adjust reserved_work_mem_mb to a bigger value.")
                else:
                    print(e, flush=True)
                import signal
                os.kill(os.getpid(), signal.SIGKILL)
        if self._thread is None:
            self._thread = Thread(target=run_wrapper)
            self._thread.start()
            return True
        return False

    def stop(self):
        if self._thread is not None:
            self._thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
