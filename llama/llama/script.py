from safetensors.torch import load_file

state_dict = {}

file = "/home/tianxin/data/LLM/TinyLlama-1.1B-Chat-v1.0/tinyllama-1b/model.safetensors"

state_dict.update(load_file(file))

count = 0
for key, val in state_dict.items():
    print(key, val)
    if count > 4:
        break
    count += 1


