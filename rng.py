import torch
import math
import numpy as np 

#set the seed for data
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True

## generates minibatch_size lower discrepancy tensors on [lower,upper]^dimension 
def Generate_sobol(size,dimension,scramble = True,device = torch.device('cuda')):
    sampler = torch.quasirandom.SobolEngine(dimension,scramble)
    m = int(math.log(size, 2))
    x = sampler.draw_base2(m).to(device)
    return x


def Normdf_inv(sample):
    v = 0.5 + (1 - torch.finfo(sample.dtype).eps) * (sample - 0.5)
    norm_sample = torch.erfinv(2 * v - 1) * math.sqrt(2)
    return norm_sample


