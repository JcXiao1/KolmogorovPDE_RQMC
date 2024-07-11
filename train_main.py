import numpy as np 
import torch
import torch.optim as optim
import os    
from tqdm import tqdm

from rng import Normdf_inv,Generate_sobol,setup_seed
from model import MLPNet,init_weights
from pdes import *

sample_type = {
                'mc':torch.rand, 
                'qmc':Generate_sobol
                }
PDES = {
        'Example1':Example1,
        'BS_Example1':BS_Example1,
        'HJB_Example1':HJB_Example1,
        'BS_Example2':BS_Example2
        }

def train_model(step,
                last_Y,
                Y,
                Z,
                config,
                pde,
                mode='start',
                save_dir=None):
    device = pde.device
    dim,N = config['dim'],config['N']
    T = pde.time_T
    delta_t = T/N
    t = step*T/N
    pde_name = config['pde']
    val_x = np.load(f'./data/{pde_name}_dim{dim}_val_x.npy')[step]
    val_y = np.load(f'./data/{pde_name}_dim{dim}_val_y.npy')[step]
    val_x = torch.Tensor(val_x).to(device)
    val_y = torch.Tensor(val_y).to(device)
    if mode == 'start':
        iter_num = config['iter_num1']
        lr = config['lr1']
        decay_rate = config['decayrate1']
        decay_step = config['decaystep1']
    elif mode == 'recur':
        iter_num = config['iter_num2']
        lr = config['lr2']
        decay_rate = config['decayrate2']
        decay_step = config['decaystep2']
    optimizer = optim.Adam(list(Y.parameters())+list(Z.parameters()), lr = lr)
    lr_reduce = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size = decay_step,
                                          gamma = decay_rate)
    loss_list = []
    err_list = []
    for i in tqdm(range(iter_num)):
        data = sample_type[config['SampleMethod']](config['batchsize'], dim+dim+dim, device=device)
        x = pde.sde(t,
                    data[:,:dim],
                    Normdf_inv(data[:,dim:(dim+dim)]))
        dw = torch.sqrt(delta_t)*Normdf_inv(data[:,(dim+dim):])
        Y.train()
        Z.train()
        optimizer.zero_grad()
        loss = torch.mean((last_Y(pde.Euler_recur(delta_t,x,dw))-pde.func_F(t,x,Y(x),Z(x),delta_t,dw))**2) #+ dim*delta_t*torch.mean((sigma*dY-Z(x))**2)
        loss.backward()
        optimizer.step()
        lr_reduce.step()
        
        Y.eval()
        err = torch.sqrt(torch.mean((Y(val_x)-val_y)**2)/torch.mean(val_y**2))
        err_list.append(err.data.to('cpu').numpy())
        loss_list.append(loss.data.to('cpu').numpy())
    if save_dir:
        step_save_dir =  os.path.join(save_dir,f'step_{step}')
        if not os.path.exists(step_save_dir):
            os.mkdir(step_save_dir)
        np.savetxt(os.path.join(step_save_dir,f'loss.txt'),loss_list)
        np.savetxt(os.path.join(step_save_dir,f'err.txt'),err_list)
        torch.save(Y,os.path.join(step_save_dir,"model_Y.pth"))
        torch.save(Z,os.path.join(step_save_dir,"model_Z.pth"))
    return Y,Z

def train(config,
          device = torch.device('cuda')):
    setup_seed(config['seed'])
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir,'config.txt'), "w") as file:    
        for parameter in config:    
            file.write(f"{parameter}: {config[parameter]}\n")

    dim,N = config['dim'],config['N']
    pde = PDES[config['pde']](dim=dim, device=device)
    T = pde.time_T
    model_Y = MLPNet(dim,1,config['width'],config['depth']).to(device)
    model_Z = MLPNet(dim,dim,config['width'],config['depth']).to(device)
    model_Y.apply(init_weights)
    model_Z.apply(init_weights)
    model_Y_prev = lambda x: pde.func_g(T,x)
    mode = 'start'
    for step in range(N-1,-1,-1):
        if step < N-1:
            mode = 'recur'
        model_Y,model_Z = train_model(step,
                                      model_Y_prev,
                                      model_Y,
                                      model_Z,
                                      config,
                                      pde,
                                      mode,
                                      save_dir)
        model_Y_prev = model_Y
        model_Y = MLPNet(dim,1,config['width'],config['depth']).to(device)
        model_Y.load_state_dict(model_Y_prev.state_dict(),strict=True)
        model_Y_prev.eval()
    
            