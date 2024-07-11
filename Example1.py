from train_main import train
import torch
import os 


    
if __name__ == '__main__':    
    config = {
                'pde': 'Example1',
                'N': 50,
                'depth': 4,
                'iter_num1':50000,
                'iter_num2':5000,
                'lr1':1e-2,
                'decaystep1': 5000,
                'decayrate1':  0.5,
                 'lr2':  1e-3,
                'decaystep2': 500,
                'decayrate2': 0.5,
                }
    device = torch.device('cuda:0')
    for bs in [2**12, 2**14, 2**16]:
        for dim in [20, 50]:
            config['dim'] = dim
            config['width'] = config['dim'] + 20
            config['batchsize'] = bs
            for seed in range(8):
                for sample in ['mc','qmc']:
                    config['seed'] = seed
                    config['SampleMethod'] = sample
                    config['save_dir'] = os.path.join(os.getcwd(),
                                                  f"{config['pde']}_dim{config['dim']}_new",
                                                  config['SampleMethod'],
                                                  f"bs_{config['batchsize']}_seed{config['seed']}_lr1_{config['lr1']}_lr2_{config['lr2']}"
                                                   )

                    train(config,device)


        
   
    
        
        
        
   