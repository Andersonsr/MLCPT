#!/bin/bash

python src/train/trainDecoder.py \
    --epochs 5 \
    --batch_size 16 \
    --dataset coco \
    --save_dir checkpoints/dinov2-opt350m \
    --decoder_name facebook/opt-350m \
    --precomputed_embeddings E:embeddings/coco/dinov2_train.pkl \
    --lora \
    --lora_rank 16 \
    --lora_alpha 32 \




