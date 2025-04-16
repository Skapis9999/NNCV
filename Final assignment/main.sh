wandb login

# python3 train.py \
#     --data-dir ./data/cityscapes \
#     --batch-size 64 \
#     --epochs 100 \
#     --lr 0.001 \
#     --num-workers 10 \
#     --seed 42 \
#     --experiment-id "unet-training" \

# python3 train_transformer.py \
#     --data-dir ./data/cityscapes \
#     --batch-size 64 \
#     --epochs 100 \
#     --lr 0.001 \
#     --num-workers 10 \
#     --seed 42 \
#     --experiment-id "attention--unet-training" \

# python3 train_transformer.py \
#     --data-dir ./data/cityscapes \
#     --batch-size 64 \
#     --epochs 100 \
#     --lr 0.0001 \
#     --num-workers 10 \
#     --seed 42 \
#     --experiment-id "attention--unet-training-pretrained-end-with-conv" \

python3 train_transformer.py \
    --data-dir ./data/cityscapes \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-128batch-512px" \