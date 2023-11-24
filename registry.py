import models
from torchvision import datasets, transforms as T
# from utils import sync_transforms as sT

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


import os
import torch
import torchvision
import torch.nn as nn 
from PIL import Image

NORMALIZE_DICT = {
    'mnist':    dict( mean=(0.1307,),                std=(0.3081,) ),
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'imagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'tinyimagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
    'cub200':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanford_dogs':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanford_cars':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_64x64': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'svhn': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'tiny_imagenet': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'imagenet_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    
    # for semantic segmentation
    'camvid': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'nyuv2': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
}


MODEL_DICT = {
    # https://github.com/polo5/ZeroShotKnowledgeTransfer
    'wrn16_1': models.wresnet.wrn_16_1,
    'wrn16_2': models.wresnet.wrn_16_2,
    'wrn40_1': models.wresnet.wrn_40_1,
    'wrn40_2': models.wresnet.wrn_40_2,

    # https://github.com/HobbitLong/RepDistiller
    'resnet8': models.resnet_tiny.resnet8,
    'resnet20': models.resnet_tiny.resnet20,
    'resnet32': models.resnet_tiny.resnet32,
    'resnet56': models.resnet_tiny.resnet56,
    'resnet110': models.resnet_tiny.resnet110,
    'resnet8x4': models.resnet_tiny.resnet8x4,
    'resnet32x4': models.resnet_tiny.resnet32x4,
    'vgg8': models.vgg.vgg8_bn,
    'vgg11': models.vgg.vgg11_bn,
    'vgg13': models.vgg.vgg13_bn,
    'vgg16': models.vgg.vgg16_bn,
    'shufflenetv2': models.shufflenetv2.shuffle_v2,
    'mobilenetv2': models.mobilenetv2.mobilenet_v2,
    
    # https://github.com/huawei-noah/Data-Efficient-Model-Compression/tree/master/DAFL
    'resnet50':  models.resnet.resnet50,
    'resnet18':  models.resnet.resnet18,
    'resnet34':  models.resnet.resnet34,
}

IMAGENET_MODEL_DICT = {
    # https://github.com/Cadene/pretrained-models.pytorch
    'resnet101': models.torchvision_models.resnet101,   
    'resnet50': models.torchvision_models.resnet50,
    'resnet34': models.torchvision_models.resnet34,
    'resnet18': models.torchvision_models.resnet18,
    'vgg11': models.torchvision_models.vgg11,
    'vgg11_bn': models.torchvision_models.vgg11_bn,
    'vgg13': models.torchvision_models.vgg13,
    'vgg13:bn': models.torchvision_models.vgg13_bn,
    'vgg16': models.torchvision_models.vgg16,
    'vgg16_bn': models.torchvision_models.vgg16_bn,
    'vgg19': models.torchvision_models.vgg19,
    'vgg19_bn': models.torchvision_models.vgg19_bn,
    'inceptionv3': models.torchvision_models.inceptionv3,
    'squeezenet1_0': models.torchvision_models.squeezenet1_0,
    'squeezenet1_1': models.torchvision_models.squeezenet1_1,   
    # 'densenet121', 'densenet169', 'densenet201', 'densenet161' and 'alexnet' are also available.
    # other models could be found in https://github.com/Cadene/pretrained-models.pytorch.
}


def get_model(name: str, num_classes, pretrained=False, **kwargs):
    if num_classes == 1000:         
        pretrained = "imagenet" if pretrained else None
        model = IMAGENET_MODEL_DICT[name](num_classes = num_classes, pretrained=pretrained)
    else:
        model = MODEL_DICT[name](num_classes=num_classes)
    return model 


def get_dataset(name: str, data_root: str='data', return_transform=False, split=['A', 'B', 'C', 'D']):
    name = name.lower()
    data_root = os.path.expanduser( data_root )

    if name=='mnist':
        num_classes = 10
        train_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])      
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.MNIST(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.MNIST(data_root, train=False, download=True, transform=val_transform)
    elif name=='cifar10':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)
    elif name=='cifar100':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
    elif name=='svhn':
        num_classes = 10
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)
    elif name=='imagenet' or name=='imagenet-0.5':
        num_classes=1000
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'ImageNet1K' )
        # train_dst = datasets.ImageNet(data_root, split='train', transform=train_transform)
        # val_dst = datasets.ImageNet(data_root, split='val', transform=val_transform)
        train_dst = torchvision.datasets.ImageFolder(
            root=os.path.join(data_root, 'train'),
            transform=train_transform)

        val_dst = torchvision.datasets.ImageFolder(
            root=os.path.join(data_root, 'val_'),
            transform=val_transform)

    elif name=='imagenet_32x32':
        num_classes=1000
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'ImageNet_32x32' ) 
        train_dst = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
    else:
        raise NotImplementedError
    if return_transform:
        return num_classes, train_dst, val_dst, train_transform, val_transform
    return num_classes, train_dst, val_dst
