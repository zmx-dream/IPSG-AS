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


        for param in self.param_groups:
            lr = param['lr']

            for p in param['params']:

                if p.grad is None:
                    continue

                p.data -= lr * p.grad.data
    
        return loss
    





