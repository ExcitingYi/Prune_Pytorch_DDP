
python prune.py --data_root data --model vgg16 --dataset cifar10 \
    --batch_size 1024 --epochs 100 --lr 0.05 --wd 5e-5 --print_freq 10 \
    --lr_decay_milestones 40,70,90 --log_tag example --prune_type irre_prune \
    --weight_file ./checkpoints/scratch/cifar10_vgg16_scratch_ddp-test.pth --retrain_epoch 50 \
    --config_file ./prune_config/vgg16_cifar10_irre.yaml --prune_freq 5 \
    --ip 127.0.0.33 --port 23456

