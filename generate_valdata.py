from pdes import *
import numpy as np
from rng import setup_seed,Normdf_inv
import os 
from tqdm import tqdm
import torch

if __name__ == '__main__':
    device = torch.device('cuda:0')
    dim_list = [20,50]
    PDES = {
            'Example1':Example1,
            'BS_Example1':BS_Example1,
            'BS_Example2':BS_Example2,
            'HJB_Example1':HJB_Example1,
           }
    for dim in dim_list:
        for pde_name in PDES:
            pde = PDES[pde_name](dim = dim, 
                                device = device)
            save_dir = f'./data'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            T = pde.time_T
            N = 50
            sample_size = 2**16
            val_x = np.zeros((N,sample_size,dim))
            val_y = np.zeros((N,sample_size,1))
            setup_seed(2024)
            for step in tqdm(range(N)):
                t = step*T/N
                data = torch.rand(sample_size,
                                dim+dim,
                                device=pde.device)
                x =  pde.sde(t, data[:,:dim], Normdf_inv(data[:,dim:]))
                y = pde.func_g(t,x)
                val_x[step] = x.to('cpu').numpy()
                val_y[step] = y.to('cpu').numpy()
            np.save(os.path.join(save_dir, f'{pde_name}_dim{dim}_val_x.npy'), val_x)
            np.save(os.path.join(save_dir, f'{pde_name}_dim{dim}_val_y.npy'), val_y)