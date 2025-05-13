rps=1.0
limit=30

conf=h_cache
alt_strategy=recomp
hcache_layers=(27)

for hcache_layer in ${hcache_layers[@]}; do
    python evaluation/bench.py \
        --model-path /home/lyx/hcache/models/Llama-2-7b-hf \
        --dataset-path evaluation/dataset/sharegpt_gpt4.json \
        --limit ${limit} \
        --rps ${rps} \
        --max-length 4096 \
        --save-path logs \
        --prefix-cache-strategy ${conf} \
        --prefix-cache-strategy-alt ${alt_strategy} \
        --hcache-layer ${hcache_layer} \
        --log-prefix sharegpt_${conf}_${hcache_layer}
done