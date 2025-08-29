#!/bin/bash

python src/train/trainDecoder.py \
    --epochs 5 \
    --batch_size 16 \
    --dataset coco \
    --save_dir checkpoints/teste \
    --decoder_name facebook/opt-350m \
    --encoder_name facebook/dinov2-base \
    --frozen_encoder \
    --logging_steps 10 \

