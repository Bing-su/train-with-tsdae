python main.py \
    --model-name="Bingsu/my_reformer_untrained" \
    --train-batch-size=16 \
    --num-epochs=1 \
    --steps-per-epoch=0 \
    --max-seq-length=16384 \
    --evaluation-steps=5000 \
    --scheduler="warmupcosinewithhardrestarts" \
    --warmup-steps=5000
