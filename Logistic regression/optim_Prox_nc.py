import random
from tqdm import tqdm
from fun_grad_nc import *
import time


# Accuracy
def accuracy(A, x, b):
    # Labels are -1, 1
    accuracy_val = ((torch.sign(torch.matmul(A, x)) == b).sum()) / len(b)
    # # Labels are 0, 1
    # accuracy_val = ((torch.round(torch.sigmoid(torch.matmul(A, x))) == b).sum()) / len(b)
    return accuracy_val

def IPSG_AS(x0, A, b, A_test, b_test, N, opts):
    x = x0
    # Initial step size
    step_size = opts['stepsize']
    # Regularization coefficient
    lambda_ = opts['lambda']
    # Maximum number of epochs
    epoch_max = opts['epoch_max']
    # Output interval
    epoch_interval = opts['epoch_interval']
    # Select the norm for regularization
    norm_q = opts['norm_q']
    # Mini-batch size
    batchsize_init = opts['batchsize_init']
    beta = opts['beta']
    step_size_init = step_size

    batchsize = batchsize_init

    de = opts['delta']
    gamma = opts['gamma']

    # Calculate initial training objective function value, test objective function value, and test accuracy
    f_train = lr_loss_f(A, b, x)
    f_test = lr_loss_f(A_test, b_test, x)
    accuracy_test = accuracy(A_test, x, b_test)
    # Store epoch iteration information
    out = {
        'fvec_train': [],  # Training set f value
        'fvec_test': [],  # Test set f value
        'epoch': [],  # Store the number of epochs for each iteration
        'accuracy_test': [],  # Test set accuracy
        'time': [],  # Time
        'stepsize': [],
        'nk': []
    }
    out['fvec_train'].append(f_train.item())
    out['fvec_test'].append(f_test.item())
    out['epoch'].append(0)
    out['accuracy_test'].append(accuracy_test.item())
    out['time'].append(0)
    out['nk'].append(batchsize)

    print('------IPSG_AS training epoch {}------'.format(0))
    print('f_train', f_train, 'f_test', f_test, 'time', 0, 'accuracy_test', accuracy_test)
    # Calculate time
    start_time = time.time()
    count = 0
    iter = 0

    y = x
    x_old = x

    for epoch in tqdm(range(1, epoch_max+1)):
        print('------IPSG_AS training epoch {}------'.format(epoch))
        total_batchsize = 0
        while total_batchsize <= N:
            out['nk'].append(batchsize)
            iter += 1
            total_batchsize += batchsize

            # Start sampling
            idx = []
            for _ in range(batchsize):
                random_num = random.randint(0, N - 1)
                idx.append(random_num)

            # Update y
            y = x + beta * (x - x_old)
            # Calculate mini-batch gradient based on sampling
            grads_y, grad_y = g_mini(A, b, y, idx)

            # Record the previous x for v update
            x_old = x
            # Proximal operator calculation
            x = prox(y - step_size * grad_y, step_size, lambda_, norm_q)

            batchsize_old = batchsize

            if iter % 10 == 0:
                # Check gradient
                vkg = grads_y - grad_y.view(-1, 1)
                # Calculate the norm along the second axis (axis=1) for each sample
                norms = torch.norm(vkg, dim=1) ** 2
                # Sum up all the norms to get the final result
                vk = torch.sum(norms) / (batchsize_old - 1)
                eps = de / (iter ** (1 + 1e-3))
                delta = eps + gamma * torch.norm(x - y) ** 2
                vkk = vk * (N - batchsize_old) / (batchsize_old * (N - 1))

                # print(vk, nk, k)
                if vkk < delta:
                    batchsize = batchsize_old
                else:
                    vk = vk / delta
                    batchsize = min(N, max(int(vk * N /(N - 1 + vk)), batchsize_old + 1))
                    step_size = min(step_size /beta, step_size_init * batchsize/batchsize_init, 2 ** 6)

        # Output results at regular intervals
        if (epoch / epoch_interval) >= count + 1:
            f_train = lr_loss_f(A, b, x)
            f_test = lr_loss_f(A_test, b_test, x)
            accuracy_test = accuracy(A_test, x, b_test)
            end_time = time.time()
            count = count + 1
            print('f_train', f_train, 'f_test', f_test, 'time', end_time - start_time, 'accuracy_test', accuracy_test, 'batchsize', batchsize, 'stepsize', step_size)
            out['fvec_train'].append(f_train.item())
            out['fvec_test'].append(f_test.item())
            out['epoch'].append(count)
            out['accuracy_test'].append(accuracy_test.item())
            out['time'].append(end_time - start_time)
    return out