import torch
from sklearn import datasets
from optim_Prox_nc import *
import time 
dataset_paths = [
("D:/codes/data/libsvm/w8a", "D:/codes/data/libsvm/w8a.t", "D:/codes/code_acc/lr_acc_nc/result/w8a/IPSG_AS_nc_1_2.txt")] 

for train_data_path, test_data_path, result_IPSG_AS in dataset_paths:
    start_time = time.time()
    print('————————————Starting setup of training set{} and test set{}———————————'.format(train_data_path, test_data_path))
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
    else:
        device = torch.device("cpu")  # Use CPU


    # Set random seed
    random_seed = 12
    torch.manual_seed(random_seed)
    # For small to medium scale, sparse matrix operations are not required

    train_dataset = datasets.load_svmlight_file(train_data_path)
    end_time_train_path = time.time()
    print('————————————Successfully read training set, Time: {}———————————'.format(end_time_train_path-start_time))
    test_dataset = datasets.load_svmlight_file(test_data_path)
    end_time_test_path = time.time()
    print('————————————Successfully read test set, Time: {}———————————'.format(end_time_test_path-start_time))

    
    # To get X in AX = b, where A is the information matrix and b is the true label
    A = torch.tensor(train_dataset[0].toarray(), dtype=torch.float32)  # Convert to a dense NumPy array X using the toarray() method
    end_time_train = time.time()
    print('————————————Training set converted to a dense matrix, Time: {}———————————'.format(end_time_train-start_time))
    A_test = torch.tensor(test_dataset[0].toarray(), dtype=torch.float32)
    end_time_test = time.time()
    print('————————————Test set converted to a dense matrix, Time: {}———————————'.format(end_time_test-start_time))
    # .unsqueeze(1) to convert to a column vector
    b = torch.tensor(train_dataset[1], dtype=torch.float32).unsqueeze(1)
    b_test = torch.tensor(test_dataset[1], dtype=torch.float32).unsqueeze(1)

    # Get variables
    N, m = torch.tensor(A.shape)
    print(N, m)
    N_test, m_test = torch.tensor(A_test.shape)
    # To ensure A and A_test have the same number of rows
    if m != m_test:
        # Create a new column of data, filled with zeros, with the same number of rows as the original tensor
        new_column = torch.zeros(N_test, m - m_test)
        # Concatenate the new column with the original tensor
        A_test = torch.cat([A_test, new_column], dim=1)

    # Initial values
    x0 = torch.randn((m, 1), dtype=torch.float32)

    print(b.size(), x0.size())

    # Import data to the device, primarily for using the GPU
    A = A.to(device)
    A_test = A_test.to(device)
    b = b.to(device)
    b_test = b_test.to(device)
    x0 = x0.to(device)

    epoch_max = 50

    norm_qs = [1/2]

    opts_IPSG_AS = {
        # Maximum number of epochs
        'epoch_max': 50,
        # Step size 
        'stepsize': 1,
        # mini_batch
        'batchsize_init': 2**2,
        # Regularization coefficient
        'lambda': 1e-2/N,
        # Select regularization term 0, 1/2, 2/3, 1
        'norm_q': 1/2,
        # Interval for outputting results
        'epoch_interval': 1,
        'beta': 0.25,
        'delta': 1,
        'gamma': 1e-3
    }

    print(opts_IPSG_AS)
    out_IPSG_AS = IPSG_AS(x0, A, b,  A_test, b_test, N, opts_IPSG_AS)
    with open(result_IPSG_AS, 'a') as file:
        file.write(f"Dataset: {str(IPSG_AS) + str(opts_IPSG_AS) +  str(out_IPSG_AS)} \n")
        file.write("=" * 40 + "\n")
        file.close()