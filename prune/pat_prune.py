import sys
import torch
import numpy as np
import collections
from itertools import permutations


""" compute density (helper fn to compute % NNZs in a tensor) """
def fill(x):
    return float(x.nonzero().size(0))/torch.numel(x)

""" m:n 1d structured best """
def pat_best(matrix, pattern):
    # Find all possible patterns.

    # Find the best m:n pattern (sum of non-masked weights).
    mask = torch.cuda.IntTensor(matrix.shape).fill_(1).view(-1,9)   # 9 = kernel_size x kernel_size
    pmax = torch.argmax(torch.matmul(matrix.abs(),pattern.t()), dim=1)
    mask[:] = pattern[pmax[:]]
    mask = mask.view(matrix.shape)
    return mask


import sys
""" returns a sparse mask """
def create_pat_mask(tensor, pattern = None):
    # Reshape tensor and mask.
    shape = tensor.shape
    ttype = tensor.type()
    t = tensor.float().contiguous()
    # sys.exit()
    # 1d-tensor
    if pattern == None:
        '''
        pattern, 4 types: 
        十，X，]，[   
        '''
        pattern = torch.tensor([[1.,0.,1.,0.,1.,0.,1.,0.,1.],[0.,1.,0.,1.,1.,1.,0.,1.,0.],[1.,1.,0.,0.,1.,0.,1.,1.,0.],[0.,1.,1.,0.,1.,0.,0.,1.,1.]])
        pattern = pattern.to(tensor)
    if len(shape) == 4:
        # 2d convs
        t = t.view(shape[0]*shape[1], shape[2]*shape[3])    # nxc, sxs
        mask = pat_best(t, pattern)
        mask = mask.contiguous()
        return mask.view(shape).type(ttype)

# 1. 统计
# 2. 生成mask



def create_kernel_mask(tensor, prune_ratio):
    # note：prune_ratio > 4/9
    if prune_ratio < 0.5:
        return torch.ones(size=tensor.shape).cuda()
    shape = tensor.shape
    ttype = tensor.type()
    t = tensor.float().contiguous()
    prune_ratio = prune_ratio - 4/9     # 还需要剪枝的比例。 4/9是pattern prune已经剪了的。
    prune_ratio = prune_ratio / (5/9)       # 忽略已经剪了的，还要剪多少kernel才能满足。
    # sys.exit()

    t = t.view(shape[0],shape[1], shape[2]*shape[3])    # nxc, sxs
    t = t.abs().sum(dim = -1)

    t_1d = t.reshape(-1)
    idx = int(t.numel() * (1-prune_ratio))
    topk = torch.topk(t_1d, idx)
    percentile=topk.values[-1]
    mask = t > percentile

    return mask.reshape(shape[0], shape[1], 1,1)


def create_mask(tensor, prune_ratio):
    mask_pat = create_pat_mask(tensor)
    mask_kernel = create_kernel_mask(tensor, prune_ratio)
    # print(mask_pat.shape)
    # print(mask_kernel.shape)
    mask = mask_pat * mask_kernel
    return mask


if __name__ == "__main__":
    temp = torch.rand(64,64,3,3).cuda()
    mask = create_kernel_mask(temp,prune_ratio=0.6)
    # print(mask)
