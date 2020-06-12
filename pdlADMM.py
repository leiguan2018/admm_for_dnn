import cupy as np
from cupy import matmul as mul
import torch.distributed as dist
import argparse
import sys
import time
import os
import socket

from util import para_func
from util.input_data import mnist, fashion_mnist, cifar10, emnist


num_classes = 10
def Net(images, label, num_classes, input_size):
    num_of_neurons = 1000
    seed_num = 13
    np.random.seed(seed=seed_num)
    W1 = np.random.normal(0, 0.1, size=(num_of_neurons, input_size))
    z1 = np.matmul(W1, images) 
    a1 = para_func.relu(z1)
    np.random.seed(seed=seed_num)
    W2 = np.random.normal(0, 0.1, size=(num_of_neurons, num_of_neurons))
    z2 = np.matmul(W2, a1) 
    a2 = para_func.relu(z2)
    np.random.seed(seed=seed_num)
    W3 = np.random.normal(0, 0.1, size=(num_classes, num_of_neurons))
    z3 = np.ones(label.shape)
    z3[label == 0] = -1
    z3[label == 1] = 1
    return W1, z1, a1, W2, z2, a2, W3, z3


# return the accuracy of the neural network model
def test_accuracy(W1, W2, W3, images, labels):
    nums = labels.shape[1]
    z1 = np.matmul(W1, images)
    a1 = para_func.relu(z1)
    z2 = np.matmul(W2, a1)
    a2 = para_func.relu(z2)
    z3 = np.matmul(W3, a2)
    # print("output shape", z3.shape, labels.shape)
    cost = para_func.cross_entropy_with_softmax(labels, z3) / nums
    pred = np.argmax(labels, axis=0)
    label = np.argmax(z3, axis=0)
    return (100.0 * np.sum(np.equal(pred, label)) / nums, cost)



def preprocess_data(x, y):
    x = np.swapaxes(x, 0, 1)
    y = np.swapaxes(y, 0, 1)
    x = np.array(x)
    y = np.array(y)

    return x, y


def admm_train(x_train, y_train, W1, z1, a1, W2, z2, a2, W3, z3, u, rho, gamma):
    W1 = para_func.update_W(W1, x_train, z1, 0, rho)
    a1 = para_func.update_a(W2, a1, z2, z1, 0, rho, gamma)
    z1 = para_func.update_z(W1, x_train, a1, z1, rho, gamma)

    W2 = para_func.update_W(W2, a1, z2, 0, rho)
    a2 = para_func.update_a(W3, a2, z3, z2, u, rho, gamma)
    z2 = para_func.update_z(W2, a1, a2, z2, rho, gamma)
    
    W3 = para_func.update_W(W3, a2, z3, u, rho)
    z3 = para_func.update_zl(W3, a2, z3, y_train, u, rho)

    u = u + rho * (z3 - mul(W3, a2))

    return W1, z1, a1, W2, z2, a2, W3, z3, u
    

def main():
    global num_classes
    size = dist.get_world_size()
    rank = dist.get_rank()

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
                        help='dataset for training (default: mnist)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--device', type=int, default=0, metavar='N',
                        help='GPU device selected (default: 0)')
    parser.add_argument('--rho', type=float, default=1, metavar='N',
                        help='rho (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=5, metavar='M',
                        help='gamma (default: 5.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()


    dataset = args.dataset
    ITER = args.epochs
    gamma = args.gamma
    rho = args.rho


    if dataset == "mnist":
        data = mnist()
        num_classes = 10
        input_size = 28 * 28
    elif dataset == "fashion_mnist":
        data = fashion_mnist()
        num_classes = 10
        input_size = 28 * 28
    elif dataset == "cifar10":
        data = cifar10()
        num_classes = 10
        input_size = 3 * 32 * 32
    elif dataset == "emnist":
        data = emnist()
        num_classes = 47
        input_size = 28*28

    print("pdlADMM. dataset={}, rho={}, gamma={}".format(dataset, rho, gamma))

    with np.cuda.Device(rank):
        x_train = data.train.xs
        y_train = data.train.ys
        x_test = data.test.xs
        y_test = data.test.ys

        data_num = x_train.shape[0]
        local_data_num = int(data_num / size)

        start, end = rank * local_data_num, (rank+1) * local_data_num

        x_train, y_train = x_train[start:end], y_train[start:end] 
        x_train, y_train = preprocess_data(x_train, y_train)
        x_test, y_test = preprocess_data(x_test, y_test)

        W1, z1, a1, W2, z2, a2, W3, z3 = Net(x_train, y_train, num_classes, input_size)

        u = np.zeros(z3.shape)
        running_time = np.zeros(ITER)
        test_acc = np.zeros(ITER)
        test_cost = np.zeros(ITER)

        for i in range(ITER):
            pre = time.time()
            W1, z1, a1, W2, z2, a2, W3, z3, u = admm_train(x_train, y_train, W1, z1, a1, W2, z2, a2, W3, z3, u, rho, gamma)
            running_time[i] = time.time() - pre

            if rank == 0:
                (test_acc[i], test_cost[i]) = test_accuracy(W1, W2, W3, x_test, y_test)
                print("epoch=", i+1, "running time:", running_time[i],
                      "test acc:", test_acc[i], "test_loss:", test_cost[i])


def init_processes(rank, size, hostname, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '25000'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn()


if __name__ == "__main__":
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    hostname = socket.gethostname()
    init_processes(world_rank, world_size, hostname, main, backend='nccl')