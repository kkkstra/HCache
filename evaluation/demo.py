import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import DeepSpeed.deepspeed as deepspeed

model_name = "/home/lyx/hcache/models/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

ds_engine = deepspeed.init_inference(
    model,
    tensor_parallel={"tp_size": 1},
    dtype=torch.float16,
)

def generate(prompt: str, max_length: int = 128):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = ds_engine.generate(
        **inputs,
        max_new_tokens=max_length,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    text = generate("你好")
    print(text)
