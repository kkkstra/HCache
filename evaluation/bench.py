import sys   
sys.setrecursionlimit(100000)

import nest_asyncio
nest_asyncio.apply()

import argparse
import asyncio
import json
import random
from typing import List
from deepspeed.inference.v2.inference_utils import PrefixCacheStrategy
import numpy as np
from transformers import AutoTokenizer

from mii.batching.data_classes import Response
import mii

class SessionItem:
    role: str
    value: str
    length: int

    def __init__(self, role: str, value: str, length: int):
        self.role = role
        self.value = value
        self.length = length

class Session:
    sid: str
    items: List[SessionItem]

    def __init__(self, sid: str, items: List[SessionItem]):
        self.sid = sid
        self.items = items

class Result:
    prompt_length: int
    generated_length: int
    ttft: float
    tbt: float

    def __init__(self, prompt_length: int, generated_length: int, ttft: float, tbt: float):
        self.prompt_length = prompt_length
        self.generated_length = generated_length
        self.ttft = ttft
        self.tbt = tbt

def setup_mii(path: str, strategy: str, strategy_alt: str, hcache_layer: int):
    def get_strategy(strategy: str):
        if strategy == "recomp":
            return PrefixCacheStrategy.RECOMP
        elif strategy == "kv_offload":
            return PrefixCacheStrategy.KV_OFFLOAD
        elif strategy == "h_cache":
            return PrefixCacheStrategy.H_CACHE
        else:
            raise ValueError(f"Unknown prefix cache strategy: {strategy}")
        
    return mii.serve(
        path,
        tensor_parallel=1,
        replica_num=1,
        prefix_cache_strategy=get_strategy(strategy),
        prefix_cache_strategy_alt=get_strategy(strategy_alt),
        h_cache_layer=hcache_layer,
    )


def demo(client):
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


def parse_args():
    parser = argparse.ArgumentParser(description="MII Client")

    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/lyx/hcache/models/Llama-2-7b-hf",
        help="Path to the model directory"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum length of the tokens"
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=1.0,
        help="Requests per second"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="logs",
        help="Path to save the results"
    )
    parser.add_argument(
        "--log-prefix",
        type=str,
        default="",
        help="Prefix for the log file"
    )
    parser.add_argument(
        "--prefix-cache-strategy",
        type=str,
        default="recomp",
        choices=["recomp", "kv_offload", "h_cache"],
        help="Prefix cache strategy to use"
    )
    parser.add_argument(
        "--prefix-cache-strategy-alt",
        type=str,
        default="h_cache",
        choices=["recomp", "kv_offload", "h_cache"],
        help="Alternative prefix cache strategy to use"
    )
    parser.add_argument(
        "--hcache-layer",
        type=int,
        default=32,
        help="Layer to use for hcache"
    )

    return parser.parse_args()


def load_dataset(model_path: str, dataset_path: str, limit: int = 0) -> List[Session]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    datas: List[Session] = []

    with open(dataset_path, "r") as f:
        sessions = json.load(f)

        count = 0
        for session in sessions:
            sid = session.get("id")
            items = session.get("items", [])

            session_items = []
            for item in items:
                role = item.get("from")
                value = item.get("value")
                tokens = tokenizer.encode(value)
                length = len(tokens)
                session_item = SessionItem(role=role, value=value, length=length)
                session_items.append(session_item)

            session_obj = Session(sid=sid, items=session_items)
            datas.append(session_obj)
            count += 1
            if limit > 0 and count >= limit:
                break

    return datas

async def generate(client, sid: str, prompt: str, prefix_length: int, max_new_tokens: int):
    response = client(prompt, sid=sid, prefix_length=prefix_length, max_new_tokens=max_new_tokens, ignore_eos=True)
    response = response[0]
    print(f"sid: {sid}, prompt_length: {response.prompt_length}, generated_length: {response.generated_length}, ttft: {response.ttft}, tbt: {response.tbt}")
    return response

async def replay_session(client, session: Session, max_length: int):
    sid = session.sid
    items = session.items

    total_length = 0
    responses = []

    prompt = ""
    prefix_length = 0
    for i in range(0, len(items), 2):
        total_length += items[i].length + items[i+1].length
        if total_length > max_length:
            break

        prompt += items[i].value
        max_new_tokens = items[i+1].length
        
        resp = await generate(client, sid, prompt, prefix_length, max_new_tokens)
        responses.append(resp)

        prompt += resp.generated_text
        prefix_length += resp.prompt_length + resp.generated_length

        await asyncio.sleep(30)
    
    return responses

async def benchmark(client, data: List[Session], max_length: int, rps: float = 1.0):
    np.random.seed(0)

    tasks = []
    for i, session in enumerate(data):
        print(f"replay session {i} {session.sid}")
        tasks.append(asyncio.create_task(replay_session(client, session, max_length)))
        await asyncio.sleep(np.random.poisson(rps))

    responses = await asyncio.gather(*tasks)

    all_responses = []
    for session_response in responses:
        all_responses.extend(session_response)

    return all_responses

def report(responses: List[Response], prefix: str, path: str, rps: float, limit: int):
    ttfts = []
    tbts = []
    result: List[Result] = []

    for response in responses:
        ttfts.append(response.ttft)
        tbts.append(response.tbt)
        result.append(Result(
            prompt_length=response.prompt_length,
            generated_length=response.generated_length,
            ttft=response.ttft,
            tbt=response.tbt
        ))

    avg_ttft = np.mean(ttfts)
    avg_tbt = np.mean(tbts)
    p99_ttft = np.percentile(ttfts, 99)
    p99_tbt = np.percentile(tbts, 99)
    p50_ttft = np.percentile(ttfts, 50)
    p50_tbt = np.percentile(tbts, 50)

    print("******************** benchmark result ********************")
    print(f"rps: {rps}")
    print(f"avg_ttft: {avg_ttft*1000:.2f} ms, p50_ttft: {p50_ttft*1000:.2f} ms, p99_ttft: {p99_ttft*1000:.2f} ms")
    print(f"avg_tbt: {avg_tbt*1000:.2f} ms, p50_tbt: {p50_tbt*1000:.2f} ms, p99_tbt: {p99_tbt*1000:.2f} ms")

    with open(f"{path}/{prefix}_{limit}_{rps}.log", "w") as f:
        f.write(f"rps: {rps}\n")
        f.write(f"avg_ttft: {avg_ttft*1000:.2f} ms, p50_ttft: {p50_ttft*1000:.2f} ms, p99_ttft: {p99_ttft*1000:.2f} ms\n")
        f.write(f"avg_tbt: {avg_tbt*1000:.2f} ms, p50_tbt: {p50_tbt*1000:.2f} ms, p99_tbt: {p99_tbt*1000:.2f} ms\n")

    # Save the response as json
    with open(f"{path}/{prefix}_{limit}_{rps}.json", "w") as f:
        json.dump(result, f, default=lambda x: x.__dict__, indent=4)


def main():
    args = parse_args()

    data = load_dataset(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        limit=args.limit
    )
    client = setup_mii(
        path=args.model_path,
        strategy=args.prefix_cache_strategy,
        strategy_alt=args.prefix_cache_strategy_alt,
        hcache_layer=args.hcache_layer
    )

    try:
        responses = asyncio.run(benchmark(client, data, max_length=args.max_length, rps=args.rps))
    except Exception as e:
        print(f"Error: {e}")


    client.terminate_server()

    report(responses,
           args.log_prefix,
           args.save_path,
           args.rps,
           args.limit)

if __name__ == "__main__":
    main()