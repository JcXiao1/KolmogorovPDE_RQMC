import torch 

class Example1():
    def __init__(self,dim = 30, device = torch.device('cuda')):
        self.time_T = torch.tensor(1).to(device)
        self.dim = torch.tensor(dim).to(device)
        self.scale = 1/torch.tensor(dim).to(device)
        self.mu = torch.tensor(0.2).to(device)*self.scale
        self.sigma =  torch.sqrt(self.scale)
        self.lower = torch.tensor(-0.5).to(device)
        self.upper = torch.tensor(0.5).to(device)
        self.device = device
        
    def sde(self,t,x,w):
        x_trans = (self.upper-self.lower)*x + self.lower
        ans = x_trans + self.mu*t + torch.sqrt(t)*self.sigma*w
        return ans 
    
    def Euler_recur(self,delta_t,x,dw):
        ans = x+ self.mu*delta_t + self.sigma*dw
        return ans 
    
    def func_f(self,t,x,y,z):
        sum_x = torch.sum(x,axis=1,keepdim=True)
        sum_z = torch.sum(z,axis=1,keepdim=True)
        T = self.time_T
        dim = self.dim
        mu = self.mu
        scale = self.scale 
        ans = (torch.cos(sum_x)+0.2*torch.sin(sum_x))*torch.exp((T-t)/2) \
                     - 0.5*(torch.sin(sum_x)*torch.cos(sum_x)*torch.exp(T-t))**2 + 0.5*scale*(y*sum_z)**2
        return ans

    def func_g(self,t,x):
        T = self.time_T
        sum_x = torch.sum(x,axis=1,keepdim=True)
        ans = torch.exp(0.5*(T-t))*torch.cos(sum_x)
        return ans 

    def func_F(self,t,x,y,z,h,w):
        return y-self.func_f(t,x,y,z)*h + torch.sum(z*w,axis = 1,keepdim=True)

class HJB_Example1():
    def __init__(self, dim = 30, device = torch.device('cuda')):
        self.time_T = torch.tensor(1).to(device)
        self.dim = torch.tensor(dim).to(device)
        self.sigma =  torch.sqrt(torch.tensor(2)).to(device)
        self.lower = torch.tensor(0).to(device)
        self.upper = torch.tensor(1).to(device)
        self.device = device
        
    def sde(self,t,x,w):
        '''
        x: [0,1]
        w: standard normal
        '''
        x_trans = (self.upper-self.lower)*x + self.lower
        ans = x_trans + torch.sqrt(t)*self.sigma*w
        return ans 
    
    def Euler_recur(self,delta_t,x,dw):
        '''
        x: [0,1]
        dw:  sqrt(delta_t)*standard normal
        '''
        ans = x + self.sigma*dw
        return ans 
    
    def func_f(self,t,x,y,z):
        ans = -torch.sum(z**2/self.sigma**2,
                         dim=1,
                         keepdim=True)
        return ans

    def func_g(self,t,x):
        T = self.time_T
        if t == T:
            ans = torch.sqrt(torch.norm(x, dim=1, keepdim=True))
        else:
            ans = torch.zeros(x.shape[0], device=self.device)
            w = torch.randn(2**20, x.shape[1], device=self.device)
            w = torch.sqrt(2*(T-t))*w
            for i in range(x.shape[0]):
                tmp = torch.exp(-torch.sqrt(torch.norm(x[i]+w, dim=1, keepdim=True)))
                tmp = -torch.log(torch.mean(tmp, dim=0))
                ans[i] = tmp
            ans = ans.unsqueeze(1)
        return ans 

    def func_F(self,t,x,y,z,h,w):
        return y-self.func_f(t,x,y,z)*h + torch.sum(z*w,axis = 1,keepdim=True)

    
class BS_Example1():
    def __init__(self, dim = 30, device = torch.device('cuda')):
        self.time_T = torch.tensor(1).to(device)
        self.dim = torch.tensor(dim).to(device)
        self.mu = torch.tensor(0.2).to(device)/self.dim
        self.sigma = 1/self.dim
        self.lower = torch.tensor(-0.5).to(device)
        self.upper = torch.tensor(0.5).to(device)
        self.device = device
        
    def sde(self,t,x,w):
        x_trans = (self.upper-self.lower)*x + self.lower
        ans = x_trans*torch.exp((self.mu - self.sigma**2/2)*t + torch.sqrt(t)*self.sigma*w)
        return ans 
    
    def Euler_recur(self,delta_t,x,dw):
        ans = x*torch.exp((self.mu - self.sigma**2/2)*delta_t + self.sigma*dw)
        return ans
    
    def func_f(self,t,x,y,z):
        sum_x = torch.sum(x,axis=1,keepdim=True)
        sum_x_2 = torch.sum(x**2,axis = 1,keepdim = True)
        sum_z = torch.sum(z,axis=1,keepdim=True)
        T = self.time_T
        dim = self.dim
        mu = self.mu
        sigma = self.sigma
        ans = (0.5*torch.cos(sum_x)+mu*sum_x*torch.sin(sum_x)+sigma**2*sum_x_2*torch.cos(sum_x))*torch.exp((T-t)/2) \
                    - 0.5/dim*(sigma*sum_x*torch.sin(sum_x)*torch.cos(sum_x)*torch.exp(T-t))**2 + 0.5/dim*(y*sum_z)**2
        return ans
    
    def func_g(self,t,x):
        T = self.time_T
        sum_x = torch.sum(x,axis=1,keepdim=True)
        ans = torch.exp(0.5*(T-t))*torch.cos(sum_x)
        return ans 
    
    def func_F(self,t,x,y,z,h,w):
        return y-self.func_f(t,x,y,z)*h + torch.sum(z*w,axis = 1,keepdim=True)
    

class BS_Example2():
    def __init__(self,dim = 30, device = torch.device('cuda')):
        self.time_T = torch.tensor(1).to(device)
        self.dim = torch.tensor(dim).to(device)
        self.mu = torch.tensor(0.2).to(device)/self.dim
        self.sigma = 1/self.dim
        self.lower = torch.tensor(-0.5).to(device)
        self.upper = torch.tensor(0.5).to(device)
        self.device = device
    
    def sde(self,t,x,w):
        x_trans = (self.upper-self.lower)*x + self.lower
        ans = x_trans*torch.exp((self.mu - self.sigma**2/2)*t + torch.sqrt(t)*self.sigma*w)
        return ans
    
    def Euler_recur(self,delta_t,x,dw):
        ans = x*torch.exp((self.mu - self.sigma**2/2)*delta_t + self.sigma*dw)
        return ans
    
    def func_f(self,t,x,y,z):
        sum_x = torch.sum(x,axis=1,keepdim=True)
        sum_x_2 = torch.sum(x**2,axis = 1,keepdim = True)
        sum_z = torch.sum(z,axis=1,keepdim=True)
        T = self.time_T
        dim = self.dim
        mu = self.mu
        sigma = self.sigma
        ans = (0.5*torch.distributions.Normal(0,1).cdf(sum_x) - mu*sum_x*torch.distributions.Normal(0,1).log_prob(sum_x).exp() \
               + sigma**2*sum_x_2*sum_x*torch.distributions.Normal(0,1).log_prob(sum_x).exp())*torch.exp((T-t)/2) \
                - 0.5/dim*(sigma*sum_x*torch.distributions.Normal(0,1).log_prob(sum_x).exp()*torch.distributions.Normal(0,1).cdf(sum_x)*torch.exp(T-t))**2 \
                    + 0.5/dim*(y*sum_z)**2
        return ans
    
    def func_g(self,t,x):
        T = self.time_T
        sum_x = torch.sum(x,axis=1,keepdim=True)
        ans = torch.exp(0.5*(T-t))*torch.distributions.Normal(0,1).cdf(sum_x)
        return ans
    
    def func_F(self,t,x,y,z,h,w):
        return y-self.func_f(t,x,y,z)*h + torch.sum(z*w,axis = 1,keepdim=True)
