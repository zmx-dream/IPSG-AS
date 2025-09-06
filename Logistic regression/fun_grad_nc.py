import torch

# Loss: differentiable part of the original problem.
def lr_loss_f(A, b, x):
    
    Ax = torch.matmul(A, x)  # Calculate the inner product
    tanh_term = torch.tanh(b * Ax)  # Calculate the tanh term
    f = (1/len(b)) * torch.sum((1 - tanh_term))

    return f


# Loss: non-differentiable part of the original problem.
def lr_loss_h(x, mu, norm_q):
    if norm_q == 1:
        h = mu * torch.norm(x, 1)
    elif norm_q == 1/2:
        h = mu * torch.norm(torch.abs(x), 1/2)

    return h

# Mini-batch of the differentiable part.
def mini_lr_loss_f(A, b, x, ind):

    bind = b[ind]
    Aind = A[ind, :]
    Ax = torch.matmul(Aind, x)  # Calculate the inner product
    tanh_term = torch.tanh(bind * Ax)  # Calculate the tanh term
    f = (1/len(ind)) * torch.sum((1 - tanh_term))

    return f


# Gradient of the differentiable part
# Full gradient
def g_batch(A, b, x):
    Ax = torch.matmul(A, x)  # Calculate the inner product
    tanh_term = torch.tanh(b * Ax)  # Calculate the tanh term
    gm = torch.mul((1 - tanh_term ** 2) * b, A).t()
    g = - (1/len(b)) * torch.sum(gm, dim=1)
    g = g.unsqueeze(1)
    
    return gm,g

# Mini-batch gradient
def g_mini(A, b, x, ind):

    bind = b[ind]
    Aind = A[ind, :]
    Ax = torch.matmul(Aind, x)
    # * for element-wise multiplication, / for element-wise division
    Ax = torch.matmul(Aind, x)  # Calculate the inner product
    tanh_term = torch.tanh(bind * Ax)  # Calculate the tanh term
    # ## Gradients for each row [g1, g2, g3]
    gm = - torch.mul((1 - tanh_term ** 2) * bind, Aind).t()
    # # Gradient, sum gives size[]
    g = (1/len(bind)) * torch.sum(gm, dim=1)
    # Add one dimension, size[,1]
    g = g.unsqueeze(1)

    return gm, g


def prox(x, lr, lambda_, norm_q):
    if norm_q == 1:
        x_abs = torch.sign(x)
        x.abs_().add_(-lr * lambda_).relu_().mul_(x_abs)

    elif norm_q == 1/2:
        # Compute threshold value
        t = (3/2) * (lr * lambda_)**(2/3)
        # Apply transformation for values greater than t
        mask = torch.abs(x) > t
        param_data_masked = x[mask]
        updated_values = (4/3) * param_data_masked * torch.cos(
            (torch.pi - torch.arccos((lr * lambda_ / 4) * (3 / torch.abs(param_data_masked)).pow(3/2))) / 3
        ).pow(2)
        x[mask] = updated_values
        # Set values less than or equal to t to 0
        x = torch.where(torch.abs(x) <= t, torch.full_like(x, 0), x)

    return x