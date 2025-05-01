from deepspeed.inference.v2.inference_utils import PrefixCacheStrategy
import mii


def setup():
    return mii.serve(
        "/home/lyx/hcache/models/Llama-2-7b-hf",
        tensor_parallel=1,
        replica_num=1,
        prefix_cache_strategy=PrefixCacheStrategy.KV_OFFLOAD,
    )


if __name__ == "__main__":
    client = setup()
    try:
        response = client("你好！", max_new_tokens=512, ignore_eos=True)
        print(f"{response=}")
    except Exception as e:
        print(f"Error: {e}")
    client.terminate_server()