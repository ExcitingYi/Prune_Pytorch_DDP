python train_scratch.py \
    --data_root data \
    --model vgg16 \
    --dataset cifar10 \
    --batch_size 1024 \
    --epochs 200 \
    --lr 0.2 \
    --wd 5e-5 \
    --print_freq 10 \
    --log_tag example