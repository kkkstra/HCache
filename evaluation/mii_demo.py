import mii


pipe = mii.pipeline("/home/lyx/hcache/models/Llama-2-7b-hf")
response = pipe(["你好！"], max_new_tokens=128)  
print(f"{response=}")
