#!/bin/bash
export USE_LIBUV=0
torchrun --nnodes 1 --nproc_per_node 1 src/train/trainDecoder.py \
    --epochs 1 \
    --batch_size 16 \
    --dataset coco \
    --save_dir checkpoints/dinov2-opt350m \
    --decoder_name facebook/opt-350m \
    --precomputed_embeddings E:embeddings/coco/dinov2_train.pkl \
    --lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --debug \




