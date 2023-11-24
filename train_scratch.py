import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# from utils.logger import get_logger
import torch.optim as optim
import torch.multiprocessing as mp
import random
import numpy as np
import registry
import utils
from utils.util import reduce_mean
from utils.validation import validate


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Basic Settings
parser.add_argument('--data_root', default='data')
parser.add_argument('--model', default='wrn40_2')
parser.add_argument('--dataset', default='cifar10')

parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--lr_decay_milestones', default="120,150,180", type=str,
                    help='milestones for learning rate decay')         
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--wd', default=5e-5, type=float,
                     help='weight decay (default: 1e-4)')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 0)')
parser.add_argument('--pretrained', action='store_true',
                    help='Use pretrained model or not')
parser.add_argument('--log_tag', default="", type=str,
                    help='log tag.')

# mp
parser.add_argument('--ip', default='127.0.0.29', type=str)
parser.add_argument('--port', default='23456', type=str)
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))

def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank
    init_seeds(local_rank+1) # set different seed for each worker

    init_method = 'tcp://' + args.ip + ':' + args.port

    ############################################
    # Initialize
    ############################################
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=args.nprocs,
                            rank=local_rank)

    ############################################
    # load data & model
    ############################################
    batch_size = int(args.batch_size / nprocs)  

    num_classes, train_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                                               sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                                              sampler=val_sampler)

    model = registry.get_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    torch.cuda.set_device(local_rank) 
    model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)     # Often works on NLP. Could be removed in CNN, cuz the batch_size is big enough. 
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])  

    ############################################
    # Initialize
    ############################################

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.wd)
    milestones = [int(ms) for ms in args.lr_decay_milestones.split(',')]
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = 'R%d-%s-%s%s'%(args.local_rank, args.dataset, args.model, args.log_tag)
    args.logger = utils.logger.get_logger(log_name, output='checkpoints/scratch/log-%s-%s%s.txt'%( args.dataset, args.model, args.log_tag))
    if args.local_rank<=0:
        for k, v in utils.logger.flatten_dict( vars(args) ).items(): # print args
            args.logger.info( "%s: %s"%(k,v) )


    ############################################
    # TODO: Checkpoints resume
    ############################################


    ############################################
    # training
    ############################################

    best_acc1 = 0
    best_ckpt = 'checkpoints/scratch/%s_%s_scratch_ddp%s.pth' % (args.dataset, args.model,args.log_tag)
    for epoch in range(args.epochs):
        start = time.time()
        args.current_epoch = epoch + 1
        model.train()
        train_sampler.set_epoch(epoch)

        for step, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            torch.distributed.barrier()
            reduced_loss = reduce_mean(loss, args.nprocs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.local_rank == 0 and args.print_freq != 0 and step % args.print_freq == 0:
                args.logger.info('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                        reduced_loss,
                        optimizer.param_groups[0]['lr'],
                        epoch=epoch+1,
                        trained_samples=step * args.batch_size + len(images),
                        total_samples=len(train_loader.dataset)
                    ))

        finish = time.time()
        if args.local_rank == 0:
            print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

        # validate after every epoch
        acc, _ = validate(test_loader, model, criterion, args)

        train_scheduler.step()

        if acc > best_acc1:
            best_acc1 = acc
            if args.local_rank == 0:
                save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.module.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer': optimizer.state_dict(),
                'scheduler': train_scheduler.state_dict()
            }, best_ckpt)

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

if __name__ == '__main__':
    main()