
import torch
from torch import nn

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128) 
        self.fc2 = nn.Linear(128, 64)   
        self.fc3 = nn.Linear(64, 10)     

    def forward(self, x):
        x = x.view(-1, 28 * 28)  
        x = self.fc1(x)
        x = x * torch.sigmoid(x)  
        x = self.fc2(x)
        x = x * torch.sigmoid(x)  
        x = self.fc3(x)
        return x
    
    
# 验证网络
if __name__ == '__main__':
    MNIST_Net = MNIST_Net()
    # 图片大小为28*28，64是一次训练的图片数量也就是minibatch=64
    input = torch.ones((1, 1, 28, 28))
    output = MNIST_Net(input)
    print(output.shape, output)

