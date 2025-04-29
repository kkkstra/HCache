import DeepSpeed-MII as mii


pipe = mii.pipeline("/home/lyx/hcache/models/Llama-2-7b-hf")
response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)  
print(response)
