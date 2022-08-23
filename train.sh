python main.py \
    --model-name="Bingsu/my_reformer_untrained" \
    --train-batch-size=16 \
    --num-epochs=100 \
    --steps-per-epoch=10000 \
    --max-seq-length=16384 \
    --evaluation-steps=5000 \
    --warmup-steps=5000
