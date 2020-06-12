# pdlADMM: An ADMM-Based Framework for Parallel Deep Learning Training with Efficiency

This repository provides the code of pdlADMM which is designed based on the ADMM framework and can effectively solve the task of fully-connected neural network problem in a data-parallel manner.

The code corresponds to our paper:
Lei Guan, Zhihui Yang, Dongsheng Li, and Xicheng Lu. pdlADMM: An ADMM-Based Framework for Parallel Deep Learning Training with Efficiency.


### Datasets included in this package
MNIST, Fashion-MNIST, CIFAR-10 and EMNIST-balanced


### Requirements
* PyTorch and Cupy: they are required to do calculations and perform communications during the training.
* Tensorflow and Keras: they are required to load the data.

### Run the code
E.g., to train the neural network on MNIST with 4 GPUs, run:
```
mpirun -np 4 python pdlADMM.py --dataset MNIST 
```

