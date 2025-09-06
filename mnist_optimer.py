import torch


class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1):
        defaults = dict(lr=lr)
        super(SGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

# 遍历参数组param_groups中的参数
        for param in self.param_groups:
            lr = param['lr']

            for p in param['params']:

                if p.grad is None:
                    continue

                p.data -= lr * p.grad.data
                # print(p.data.shape, p.grad.data.shape)


        return loss
    

class psg(torch.optim.Optimizer):

    def __init__(self, params, lambda_reg=1e-6, step_size=1e-2):
        defaults = dict(lambda_reg=lambda_reg, step_size=step_size)
        super(psg, self).__init__(params, defaults)
    
    def sgd(self):
        for group in self.param_groups:
            step_size = group['step_size']
            for param in group['params']:
                if param.grad is not None:
                    param.data -= step_size * param.grad.data
                
    def prox_l1(self):
        for group in self.param_groups:
            lambda_reg = group['lambda_reg']
            step_size = group['step_size']
            for param in group['params']:
                if param.grad is not None:
                    param.data = torch.sign(param.data) * torch.max(torch.abs(param.data) - lambda_reg * step_size, torch.zeros_like(param.data))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # 首先应用梯度下降
        self.sgd()

        # 然后应用软阈值
        self.prox_l1()

        return loss
    

    
# 函数 h(x)=μ∥x∥1 对应的邻近算子 sign(x)max{|x|−μ,0}。
def prox_l1(x, mu):
    # y = torch.max(torch.abs(x) - mu, torch.tensor(0))
    # y = torch.sign(x) * y
    x_abs = torch.sign(x)
    x.abs_().add_(-mu).relu_().mul_(x_abs)
    return x





