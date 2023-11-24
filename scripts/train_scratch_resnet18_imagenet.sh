python train_scratch.py \
    --data_root data \
    --model resnet18 \
    --dataset imagenet \
    --batch_size 1024 \
    --epochs 200 \
    --lr 0.05 \
    --wd 5e-5 \
    --print_freq 10 \
    --log_tag example
