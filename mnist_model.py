# 搭建神经网络, MNIST是个10分类的测试集
import torch
from torch import nn


# class MNIST_Net(nn.Module):
#     def __init__(self):
#         super(MNIST_Net, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 128)  # 28*28到128
#         self.fc2 = nn.Linear(128, 64)    # 隐藏层到另一个隐藏层
#         self.fc3 = nn.Linear(64, 10)     # 输出层，因为MNIST是十分类问题

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)  # 将图像数据展平
#         x = torch.relu(self.fc1(x))  
#         x = torch.relu(self.fc2(x))  
#         x = self.fc3(x)
#         return x
    
# class MNIST_Net(nn.Module):
#     def __init__(self):
#         super(MNIST_Net, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 128)  # 28*28到128
#         self.fc2 = nn.Linear(128, 64)    # 隐藏层到另一个隐藏层
#         self.fc3 = nn.Linear(64, 10)     # 输出层，因为MNIST是十分类问题

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)  # 将图像数据展平
#         x = torch.sigmoid(self.fc1(x))  
#         x = torch.sigmoid(self.fc2(x))  
#         x = self.fc3(x)
#         return x

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 28*28到128
        self.fc2 = nn.Linear(128, 64)    # 隐藏层到另一个隐藏层
        self.fc3 = nn.Linear(64, 10)     # 输出层，因为MNIST是十分类问题

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将图像数据展平
        x = self.fc1(x)
        x = x * torch.sigmoid(x)  
        x = self.fc2(x)
        x = x * torch.sigmoid(x)  
        x = self.fc3(x)
        return x
    

class Swish(nn.Module):
  def forward(self, input):
    return (input * torch.sigmoid(input))
  
  def __repr__(self):
    return self.__class__.__name__ + ' ()'

class CNNModel(nn.Module):
    def __init__ (self):
        super(CNNModel,self).__init__()
        
        self.cnn1=nn.Conv2d(in_channels=1,out_channels=16, kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(16)
        self.swish1=Swish()
        nn.init.xavier_normal_(self.cnn1.weight)
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=1)
        
        self.cnn2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(32)
        self.swish2=Swish()
        nn.init.xavier_normal_(self.cnn2.weight)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        
        self.cnn3=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(64)
        self.swish3=Swish()
        nn.init.xavier_normal_(self.cnn3.weight)
        self.maxpool3=nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64*6*6,10)
        
        
    def forward(self,x):
        out=self.cnn1(x)
        out=self.bn1(out)
        out=self.swish1(out)
        out=self.maxpool1(out)
        out=self.cnn2(out)
        out=self.bn2(out)
        out=self.swish2(out)
        out=self.maxpool2(out)
        out=self.cnn3(out)
        out=self.bn3(out)
        out=self.swish3(out)
        out=self.maxpool3(out)
        out=out.view(out.size(0),-1)
        out=self.fc1(out)

        return out
    
# 验证网络
if __name__ == '__main__':
    MNIST_Net = CNNModel()
    # 图片大小为28*28，64是一次训练的图片数量也就是minibatch=64
    input = torch.ones((1, 1, 28, 28))
    output = MNIST_Net(input)
    print(output.shape, output)

