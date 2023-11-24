
python prune.py --data_root data --model resnet18 --dataset imagenet \
    --batch_size 1024 --epochs 100 --lr 0.01 --wd 5e-5 --print_freq 10 \
    --lr_decay_milestones 40,70,90 --log_tag example --prune_type irre_prune \
    --weight_file ./checkpoints/pretrained/resnet18_imagenet.pth --retrain_epoch 50 \
    --config_file ./prune_config/resnet18_imagenet_irre.yaml --prune_freq 5 \
    --ip 127.0.0.25 --port 23444

