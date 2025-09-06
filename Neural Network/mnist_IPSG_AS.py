import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from mnist_optimer import *
from mnist_model import *
from torch.utils.data import DataLoader
import time 

# Define the training device, "cuda:0" is the first GPU, or you can use "cuda" directly
device = torch.device("cuda")
# Set the random seed
torch.manual_seed(42)   # The main random seed
torch.cuda.manual_seed_all(42)   # If using CUDA, set the random seed for all GPUs

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])

train_data = torchvision.datasets.MNIST(root='D:/codes/Gradient_Sampling_LineSeach/GSL_NN/MNIST/datasets/mnist_dataset', train=True, download=True, transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.MNIST(root='D:/codes/Gradient_Sampling_LineSeach/GSL_NN/MNIST/datasets/mnist_dataset', train=False, download=True, transform=torchvision.transforms.ToTensor())

# train_data = torchvision.datasets.FashionMNIST(root='D:/codes/Gradient_Sampling_LineSeach/GSL_NN/MNIST/datasets/mnist_dataset', train=True, download=True, transform=torchvision.transforms.ToTensor())

# test_data = torchvision.datasets.FashionMNIST(root='D:/codes/Gradient_Sampling_LineSeach/GSL_NN/MNIST/datasets/mnist_dataset', train=False, download=True, transform=torchvision.transforms.ToTensor())

train_images_list = []
train_labels_list = []
for i in tqdm(range(len(train_data))):
    img, label = train_data[i]
    train_images_list.append(img)
    train_labels_list.append(label)

train_images_tensor = torch.stack(train_images_list).to(device)
train_labels_tensor = torch.tensor(train_labels_list).to(device)

# Length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("Training dataset length: {}".format(train_data_size))
print("Test dataset length: {}".format(test_data_size))


# Instantiate the network
model = MNIST_Net()
model_name = 'MNIST'
data_name = 'mnist'
# Transfer the network model to the device
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Transfer the loss function to cuda
criterion = criterion.to(device)

step_sizes =  [2 ** (-2)]
batch_sizes = [2**6]  
epss = [1000]
gammas = [1e-3]
betas = [0.25]

# Save the initial neural network information
model_init_save = copy.deepcopy(model.state_dict())
for beta in betas:
    for  lr_init in step_sizes:
        for  gamma in gammas:
            for  eps in epss:
                # Initialize before each training session
                model.load_state_dict(model_init_save)
                opt = {
                    'train_data_size': train_data_size,
                    'test_data_size': test_data_size,
                    'lambda_reg': 1e-2 / train_data_size,
                    'batch_size': 2**6,
                    'n_epochs': 50,
                    'lr_init': lr_init,
                    'lr_max': 1,
                    'norm_q': 1/2,
                    'beta': beta,
                    'eps_0': eps,
                    'gamma_1': gamma
                }
                # Save results
                result ={ 
                    'train_loss_val': [],
                    'test_loss_val': [],
                    'train_loss_reg_val': [],
                    'test_loss_reg_val': [],
                    'train_acc': [],
                    'test_acc': [],
                    'time': [],
                    'lr': [],
                    'nk': []
                }
                print('Parameters', opt)
                test_loss_min = np.Inf # track change in validation loss

                lambda_reg = opt['lambda_reg']
                norm_q = opt['norm_q']

                # Initial batch size
                batch_size = opt['batch_size']
                batch_size_init = batch_size

                # Number of epochs
                n_epochs = opt['n_epochs']

                # Step size and maximum step size
                lr_init = opt['lr_init']
                lr_max = opt['lr_max']

                # Momentum coefficient
                beta = opt['beta']
                # Dynamic sampling coefficient
                eps_0 = opt['eps_0']
                gamma_1 = opt['gamma_1']


                # Record the number of training iterations
                total_train_step = 0
                # Record the number of testing iterations
                total_test_step = 0


                # Number of training iterations
                start_time = time.time()
                batch_size_loader = batch_size
                # Use DataLoader to load the dataset
                train_dataloader = DataLoader(train_data, batch_size=batch_size_loader)
                test_dataloader = DataLoader(test_data, batch_size=batch_size_loader)
                # Use DataLoader to load the dataset for model validation
                batch_size_loader_acc = 500
                train_dataloader_acc = DataLoader(train_data, batch_size=batch_size_loader_acc)
                test_dataloader_acc = DataLoader(test_data, batch_size=batch_size_loader_acc)

                # Calculate the initial function value and accuracy
                loss_reg = 0.0
                train_loss = 0.0
                test_loss = 0.0
                train_total_sample = 0
                train_right_sample = 0
                test_total_sample = 0
                test_right_sample = 0

                for data, target in train_dataloader_acc:
                    data = data.to(device)
                    target = target.to(device)
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = model(data).to(device)
                    # calculate the batch loss
                    loss = criterion(output, target)
                    # update average validation loss 
                    train_loss += loss.item()*data.size(0)
                    # convert output probabilities to predicted class
                    _, pred = torch.max(output, 1)    
                    # compare predictions to true label
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    # correct = np.squeeze(correct_tensor.to(device).numpy())
                    train_total_sample += batch_size_loader_acc
                    for i in correct_tensor:
                        if i:
                            train_right_sample += 1
                
                for data, target in test_dataloader_acc:
                    data = data.to(device)
                    target = target.to(device)
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = model(data).to(device)
                    # calculate the batch loss
                    loss = criterion(output, target)
                    # update average validation loss 
                    test_loss += loss.item()*data.size(0)
                    # convert output probabilities to predicted class
                    _, pred = torch.max(output, 1)    
                    # compare predictions to true label
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    # correct = np.squeeze(correct_tensor.to(device).numpy())
                    test_total_sample += batch_size_loader_acc
                    for i in correct_tensor:
                        if i:
                            test_right_sample += 1
                print("~Train_Accuracy:",100*train_right_sample/train_total_sample,"%", "Test_Accuracy:",100*test_right_sample/test_total_sample,"%")

                # Calculate average loss
                if norm_q == 1:
                    for param in model.parameters():
                        loss_reg += torch.sum(torch.abs(param.data))
                elif norm_q == 1/2:
                    for param in model.parameters():
                        loss_reg += torch.sum(torch.sqrt(torch.abs(param.data)))

                train_loss = train_loss/len(train_dataloader_acc.sampler) 
                test_loss = test_loss/len(test_dataloader_acc.sampler)

                # Display the loss and learning rate for the training and validation sets
                print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Loss+reg: {:.6f} \ttest Loss: {:.6f} \ttest Loss+reg: {:.6f} \tbatch_size: {:.6f},  \tlr: {:.6f}'.format(
                    0, train_loss, train_loss + lambda_reg * loss_reg.item(), test_loss, test_loss + lambda_reg * loss_reg.item(), batch_size, lr_init))

                result['train_loss_val'].append(train_loss)
                result['test_loss_val'].append(test_loss)
                result['train_loss_reg_val'].append(train_loss + lambda_reg * loss_reg.item())
                result['test_loss_reg_val'].append(test_loss + lambda_reg * loss_reg.item())
                result['train_acc'].append(train_right_sample/train_total_sample)
                result['test_acc'].append(test_right_sample/test_total_sample)
                result['lr'].append(lr_init)

                k = 0
                optimizer = SGD(model.parameters(),  lr = lr_init)
                lr = lr_init
                
                x_old =  copy.deepcopy([param.data for param in model.parameters()])

                deleta_x = torch.zeros(1).to(device)

                for epoch in tqdm(range(1, n_epochs+1)):
                    print("--------Epoch {} training starts--------".format(epoch))

                

                    total_batchsize = 0
                    while total_batchsize < train_data_size:
                        result['nk'].append(batch_size)
                        k += 1
                        total_batchsize += batch_size
                        # y_k = x_k + \beta (x_k - x_{k-1})
                        for i, param in enumerate(model.parameters()):
                            param.data = param.data + beta * (param.data - x_old[i])

                        # Calculate the gradient at y_k
                        indices = torch.randint(0, train_data_size, (batch_size,), device=device)
                        stacked_images = train_images_tensor[indices]
                        stacked_targets = train_labels_tensor[indices]
                    
                        # The original code remains unchanged
                        optimizer.zero_grad()
                        outputs = model(stacked_images)
                        loss = criterion(outputs, stacked_targets)
                        loss.backward()

                        grad_y = [param.grad.data.clone() for param in model.parameters()]

                        x_old =  copy.deepcopy([param.data for param in model.parameters()])
                        # x_{k+1} = prox(y_k - lr * grad(y_k))    
                        for i, param in enumerate(model.parameters()):
                            param.data = param.data - lr * grad_y[i]

                        # Proximal operator update
                        if norm_q == 1/2:
                            for i, param in enumerate(model.parameters()):
                                # Compute threshold value
                                t = (3/2) * (lr * lambda_reg)**(2/3)
                                # Change values greater than t
                                mask = torch.abs(param.data) > t
                                param_data_masked = param.data[mask]
                                updated_values = (4/3) * param_data_masked * torch.cos(
                                    (torch.pi - torch.arccos((lr * lambda_reg / 4) * (3 / torch.abs(param_data_masked)).pow(3/2))) / 3
                                ).pow(2)
                                param.data[mask] = updated_values
                                # Set values less than or equal to t to 0
                                param.data = torch.where(torch.abs(param.data) <= t, torch.full_like(param.data, 0), param.data)

                        elif norm_q == 1:
                            for i, param in enumerate(model.parameters()):
                                     param.data = torch.sign(param.data) * torch.max(torch.abs(param.data) - lambda_reg * lr, torch.zeros_like(param.data))

                        if k % 100 == 0:
                            v_k = torch.zeros(1).to(device)
                            delta_x_old = deleta_x

                            delta_x = torch.zeros(1).to(device)
                            for i, param in enumerate(model.parameters()):
                                delta_x += torch.sum(torch.mul(param.data - x_old[i], param.data - x_old[i])) 

                            eps = eps_0 / ((k) ** (1.001)) 
                            delta = eps + gamma_1 * delta_x 

                            
                            for idx in range(len(stacked_targets)):
                                image = stacked_images[idx].unsqueeze(0)
                                target = stacked_targets[idx].unsqueeze(0)
                                optimizer.zero_grad()
                                output = model(image)
                                loss = criterion(output, target)
                                loss.backward()
                    
                                for i, param in enumerate(model.parameters()):
                                    v_k += torch.sum(torch.mul(param.grad - grad_y[i], param.grad - grad_y[i]))
                    

                            v_k = v_k / (batch_size - 1)

                            if v_k * (train_data_size - batch_size) / (batch_size * (train_data_size - 1)) < delta:
                                batch_size = batch_size
                            else:
                                v_k_delta = v_k / delta
                                print(v_k, train_data_size, v_k_delta * train_data_size / (train_data_size - 1 + v_k_delta), batch_size + 1)
                                batch_size = min(train_data_size, max(int(v_k_delta * train_data_size / (train_data_size - 1 + v_k_delta)), batch_size + 1)) 
                                lr = min(lr/beta, lr_init * batch_size / batch_size_init, lr_max)

                        
                    # Calculate function value and accuracy
                    loss_reg = 0.0
                    train_loss = 0.0
                    test_loss = 0.0
                    train_total_sample = 0
                    train_right_sample = 0
                    test_total_sample = 0
                    test_right_sample = 0
                
                    for data, target in train_dataloader_acc:
                        data = data.to(device)
                        target = target.to(device)
                        # forward pass: compute predicted outputs by passing inputs to the model
                        output = model(data).to(device)
                        # calculate the batch loss
                        loss = criterion(output, target)
                        # update average validation loss 
                        train_loss += loss.item()*data.size(0)
                        # convert output probabilities to predicted class
                        _, pred = torch.max(output, 1)    
                        # compare predictions to true label
                        correct_tensor = pred.eq(target.data.view_as(pred))
                        # correct = np.squeeze(correct_tensor.to(device).numpy())
                        train_total_sample += batch_size_loader_acc
                        for i in correct_tensor:
                            if i:
                                train_right_sample += 1
                
                    for data, target in test_dataloader_acc:
                        data = data.to(device)
                        target = target.to(device)
                        # forward pass: compute predicted outputs by passing inputs to the model
                        output = model(data).to(device)
                        # calculate the batch loss
                        loss = criterion(output, target)
                        # update average validation loss 
                        test_loss += loss.item()*data.size(0)
                        # convert output probabilities to predicted class
                        _, pred = torch.max(output, 1)    
                        # compare predictions to true label
                        correct_tensor = pred.eq(target.data.view_as(pred))
                        # correct = np.squeeze(correct_tensor.to(device).numpy())
                        test_total_sample += batch_size_loader_acc
                        for i in correct_tensor:
                            if i:
                                test_right_sample += 1
                    print("~Train_Accuracy:",100*train_right_sample/train_total_sample,"%", "Test_Accuracy:",100*test_right_sample/test_total_sample,"%")


                    # Regularization term loss
                    if norm_q == 1:
                        for param in model.parameters():
                            loss_reg += torch.sum(torch.abs(param.data))
                    elif norm_q == 1/2:
                        for param in model.parameters():
                            loss_reg += torch.sum(torch.sqrt(torch.abs(param.data)))

                    train_loss = train_loss/len(train_dataloader_acc.sampler) 
                    test_loss = test_loss/len(test_dataloader_acc.sampler)

                    # Display the loss and learning rate for the training and validation sets
                    print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Loss+reg: {:.6f} \ttest Loss: {:.6f} \ttest Loss+reg: {:.6f} \tbatch_size: {:.6f},  \tlr: {:.6f}'.format(
                        epoch, train_loss, train_loss + lambda_reg * loss_reg.item(), test_loss, test_loss + lambda_reg * loss_reg.item(), batch_size, lr))
                    
                    result['train_loss_val'].append(train_loss)
                    result['test_loss_val'].append(test_loss)
                    result['train_loss_reg_val'].append(train_loss + lambda_reg * loss_reg.item())
                    result['test_loss_reg_val'].append(test_loss + lambda_reg * loss_reg.item())
                    result['train_acc'].append(train_right_sample/train_total_sample)
                    result['test_acc'].append(test_right_sample/test_total_sample)
                    result['lr'].append(lr)


                    end_time = time.time()
                    total_time = end_time - start_time
                    result['time'].append(total_time)


                    total_test_step = total_test_step + 1


                with open('D:/codes/code_acc_KL/code/nonconvex_NN/result_mnist/result_mnist_Prox_IPSG_rand.txt', 'a') as file:
                    file.write(f"{'mnist_PPA_Swish'} Dataset:  {str(opt)} \n" + model_name + data_name)
                    file.write(str(result) + '\n')
                    file.write("=" * 40 + "\n")
                    file.close()