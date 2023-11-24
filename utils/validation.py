import time
import torch
import torch.distributed as dist
from utils.util import reduce_mean, AverageMeter, ProgressMeter, accuracy


def validate(val_loader, model, criterion, args, print_log = True):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if local_rank == 0:
            #     progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        if args.local_rank == 0 and print_log:
            args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f}'
                .format(current_epoch=args.current_epoch, acc1=top1.avg, acc5=top5.avg, loss=losses.avg))

    return top1.avg, top5.avg

import numpy as np
import torch.nn as nn



class Sparsity_hook(object):
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.batch_sum = 0
        self.sparsity = 0

    def hook_fn(self, module, input, output):
        zeros = torch.sum(output == 0)
        nonzeros = torch.sum(output != 0)
        local_sparsity = zeros/(zeros+nonzeros)
        all_sparsity = (self.sparsity * self.batch_sum + local_sparsity)
        self.batch_sum += 1
        self.sparsity = all_sparsity / self.batch_sum


    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module)


def test_acti_sparsity_ddp(val_loader, model, criterion, local_rank, args, print_log = True):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.ReLU):
            hooks.append([name,Sparsity_hook(m)])
    # switch to evaluate mode

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if local_rank == 0:
            print('Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f}'
                .format(acc1=top1.avg, acc5=top5.avg, loss=losses.avg))
            # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    for ls in hooks:
        mean_sparsity = reduce_mean(ls[1].sparsity, args.nprocs)
        if local_rank == 0:
            print(ls[0], "Output Sparsity: ", mean_sparsity, "wo_mean", ls[1].sparsity)

    return top1.avg, top5.avg