
import torch

def magnitude_mask(tensor, prune_ratio):
    t = tensor.abs().sum((1,2,3))
    t_1d = t.reshape(-1)
    idx = int(t.numel() * (1-prune_ratio))
    topk = torch.topk(t_1d, idx)
    percentile=topk.values[-1]
    # percentile = np.percentile(t, prune_ratio)
    # percentile = torch.tensor(percentile).cuda()
    mask = t > percentile
    return mask.reshape(-1,1,1,1)


""" returns a sparse mask """
def create_mask(tensor, prune_ratio=0.5):
    mask = magnitude_mask(tensor, prune_ratio)
    return mask

# torch.ones(self.out_channels, self.kernel_size[0], self.kernel_size[1])
# self.weight * self.FilterSkeleton.unsqueeze(1)
if __name__ == "__main__":
    t = torch.rand(512,256,3,3).cuda()
    masks = create_mask(t, prune_ratio=0.9)
    print(masks.shape)
