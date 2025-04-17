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

# python3 train_transformer.py \
#     --data-dir ./data/cityscapes \
#     --batch-size 64 \
#     --epochs 100 \
#     --lr 0.0001 \
#     --num-workers 10 \
#     --seed 42 \
#     --experiment-id "attention--unet-training-pretrained-end-with-conv-scheduler-freeze-and-unfreeze-64batch-512px" \

# python3 transfer_learning.py \
#     --data-dir ./data/cityscapes \
#     --batch-size 64 \
#     --epochs 100 \
#     --lr 0.0001 \
#     --num-workers 10 \
#     --seed 42 \
#     --experiment-id "sam-vit-h-transfer" \
#     --sam-checkpoint sam_vit_h_4b8939.pth

# python3 train_light.py \
#     --data-dir ./data/cityscapes \
#     --batch-size 64 \
#     --epochs 100 \
#     --lr 0.01 \
#     --num-workers 10 \
#     --seed 42 \
#     --experiment-id "BowlNet_64_batch_512px"

python3 train_afformer.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "afformer-tiny"

