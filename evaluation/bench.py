from deepspeed.inference.v2.inference_utils import PrefixCacheStrategy
import mii


def setup():
    return mii.serve(
        "/home/lyx/hcache/models/Llama-2-7b-hf",
        tensor_parallel=1,
        replica_num=1,
        prefix_cache_strategy=PrefixCacheStrategy.H_CACHE,
        prefix_cache_strategy_alt=PrefixCacheStrategy.RECOMP,
    )


if __name__ == "__main__":
    client = setup()
    try:
        prompt = "你好"
        sid = "test"

        response = client(prompt, sid=sid, max_new_tokens=1024, ignore_eos=True)
        print(f"{response[0].prompt_length=} {response[0].generated_length=}")

        prompt += response[0].generated_text
        response = client(prompt, sid=sid, max_new_tokens=1024, ignore_eos=True)
        print(f"{response[0].prompt_length=} {response[0].generated_length=}")
    except Exception as e:
        print(f"Error: {e}")
    client.terminate_server()