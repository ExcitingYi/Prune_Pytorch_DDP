
import torch

def magnitude_mask(tensor, prune_ratio):
    t = tensor.abs()
    t_1d = t.reshape(-1)
    idx = int(t.numel() * (1-prune_ratio))
    topk = torch.topk(t_1d, idx)
    percentile=topk.values[-1]
    mask = t > percentile
    return mask


""" returns a sparse mask """
def create_mask(tensor, prune_ratio=0.5):

    mask = magnitude_mask(tensor, prune_ratio)

    return mask

if __name__ == "__main__":
    t = torch.rand(512,256,3,3).cuda()
    masks = create_mask(t, prune_ratio=0.9)
    print(masks.shape)
    t_ = t * masks

    print(t_[0])
