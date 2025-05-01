import mii

if __name__ == "__main__":
    client = mii.client("/home/lyx/hcache/models/Llama-2-7b-hf")
    client.terminate_server()