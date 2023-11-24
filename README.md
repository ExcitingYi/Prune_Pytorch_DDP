# DNN Model Prune - Pytorch-DDP

A naive implementation of DNN model prune. The project re-implemented some popular prune methods. 
1. Filter prune 
2. [Stripe prune](https://proceedings.neurips.cc/paper/2020/hash/ccb1d45fb76f7c5a0bf619f979c6cf36-Abstract.html)
3. [PatDNN](https://arxiv.org/pdf/2001.00138.pdf)
4. [M:N prune](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html)
5. [irregular](https://arxiv.org/pdf/1510.00149.pdf)

## 1. Quik Start

### 1.1 Training From Scratch

Train VGG-16 on CIFAR-10:
```
python train_scratch.py --data_root data --model vgg16 --dataset cifar10 --batch_size 1024 \
    --epochs 200 --lr 0.2 --wd 5e-5 --print_freq 10 --log_tag example
```


For single GPU Training, the batch_size and learning rate should be adjusted.  
In this repo, various of models are provided, including wide-resnet, resnet for CIFAR, resnet for ImageNet, VGG, Inception-V3 and so on. 

### 1.2 Prune and Fine-tune
Irregular prune on VGG-16, CIFAR-10
```
python prune.py --data_root data --model vgg16 --dataset cifar10 \
    --batch_size 1024 --epochs 100 --lr 0.05 --wd 5e-5 --print_freq 10 \
    --lr_decay_milestones 40,70,90 --log_tag example --prune_type irre_prune \
    --weight_file ./checkpoints/scratch/cifar10_vgg16_scratch_ddp-test.pth --retrain_epoch 50 \
    --config_file ./prune_config/vgg16_cifar10_irre.yaml --prune_freq 5 
```

The `--weight_file` is the pretrained VGG-16 model where you saved. But for CIFAR-10, it is not necessary to provied, cuz it's easy to train. But for ImageNet, the pretrained models should be provided. You can also set `--pretrained True ` to use the pretrained torchvision models. The `--config_file` is the prune ratio setup. 


More details could be found in the file [./scripts/](./scripts/)

## 2. Results
Some results of VGG16 CIFAR-10 could be found in [./checkpoints](./checkpoints/). We list the performance of ResNet18 and VGG16 on the ImageNet dataset. 

<table>
    <tr>
        <td>Model</td> 
        <td>Prune Method</td> 
        <td>Prune ratio</td>
        <td>Acc@1</td> 
   </tr>
    <tr>
        <td rowspan="4">VGG16</td>    
  		 <td>irregular</td> 
         <td>79.26</td>
      	 <td>71.59</td> 
    </tr>
    <tr>
        <td>PatDNN</td> 
        <td>74.55*</td>   
        <td>70.88</td>
    </tr>
    <tr>
        <td>m4n2</td>
        <td>48.52</td>
        <td>71.74</td> 
    </tr>
    <tr>
        <td>Stripe</td>
        <td>45.59*</td>
        <td>71.48</td>
    </tr>
    <tr>
        <td rowspan="4">ResNet18</td>    
  		 <td>irregular</td> 
         <td>73.13</td>
      	 <td>69.75</td> 
    </tr>
    <tr>
        <td>PatDNN</td> 
        <td>68.99*</td>   
        <td>69.18</td>
    </tr>
    <tr>
        <td>m4n2</td>
        <td>47.01</td>
        <td>70.31</td> 
    </tr>
    <tr>
        <td>Stripe</td>
        <td>44.02*</td>
        <td>69.74</td>
    </tr>

</table>

The pruning ratio with star* is the conv_layer pruning ratio, excluding FC layer. As the pruning method could only be applied in conv layer. 

## 3. Pruning granularity

Generally, for the same training process, fine-grained pruning could get higher prune ratios and higher performance than course-grained pruning. 
Here are the different prune granularity listed in [Pruning filter in fitler](https://proceedings.neurips.cc/paper/2020/hash/ccb1d45fb76f7c5a0bf619f979c6cf36-Abstract.html).

![prune_granularity1](./imgs/different%20grain.png#pic_center)

For PatDNN, it is combined by connectivity pruning and pattern pruning. 

![patDNN](./imgs/patdnn.png#pic_center)


Note1. The code cannot achieve sota performance. This is because our pruning method simply uses the absolute values of weights as the criterion for pruning, witch is also called magnitude pruning. If you want to get higher prune ratio and performance, it is recommended to refer to [ADMM-NN](https://dl.acm.org/doi/abs/10.1145/3297858.3304076), [Movement Pruning](https://proceedings.neurips.cc/paper/2020/hash/eae15aabaa768ae4a5993a8a4f4fa6e4-Abstract.html), and other related works. 

Note2. The pruned model in this repo cannot be accelerated by GPU directly. If you want to accelerate the inference in Pytorch, you could use the stripe prune and filter prune and refer to this [repo](https://github.com/fxmeng/Pruning-Filter-in-Filter), which is also the source code of stripe pruning. Besides, set the prune ratio to "m4n2" in mn prune, the pruned model could be accelerated with Ampere arch GPUs (RTX30, A100...) using TensorRT. The NVIDIA official tutorial is in [here](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/). 



## 4. References

Han S, Mao H, Dally W J. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding[J]. arXiv preprint arXiv:1510.00149, 2015.

Pool J, Yu C. Channel permutations for N: M sparsity[J]. Advances in neural information processing systems, 2021, 34: 13316-13327.

Niu W, Ma X, Lin S, et al. Patdnn: Achieving real-time dnn execution on mobile devices with pattern-based weight pruning[C]//Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems. 2020: 907-922.

Meng F, Cheng H, Li K, et al. Pruning filter in filter[J]. Advances in Neural Information Processing Systems, 2020, 33: 17629-17640.

Sanh V, Wolf T, Rush A. Movement pruning: Adaptive sparsity by fine-tuning[J]. Advances in Neural Information Processing Systems, 2020, 33: 20378-20389.

Ren A, Zhang T, Ye S, et al. Admm-nn: An algorithm-hardware co-design framework of dnns using alternating direction methods of multipliers[C]//Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems. 2019: 925-938.


