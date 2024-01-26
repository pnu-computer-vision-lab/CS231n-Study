# coding: utf-8
import sys, os
sys.path.append(os.pardir)

# Batch Normalization
# -------------------------------------------------------------------------------------
# Force the activations throughout a network to take on a unit "gaussian distribution"

# Better training stage performance
# Not depending on the initialized weight
# Suppress the overfitting

import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

# Read the MNIST data-set

# Normalize the image's pixel values into the range [0.0, 1.0]
# Flatten the input image into 1-D array

# (train image, train label), (test image, test label)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Truncate the training data
x_train = x_train[:1000]
t_train = t_train[:1000]

# Setting some parameter values
max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def __train(weight_init_std):
    """
    Perform the training stage,
    as well as comparing the results between whether the batch normalization has taken.
    (ie. Train two neural networks)
    
    Params:
        weight_init_std: given initialized weight's standard deviation
        
    Return:
        train_acc_list:
            a list of accuracy for the network without batch normalization
        bn_train_acc_list:
            a list of accuracy for the network with batch normalization
    """
    
    # Fully-Connected Neural Network

    # Train with using the batch normalization
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100],
                                    output_size=10, weight_init_std=weight_init_std, use_batchnorm=True)
    
    # Without the batch normalization
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100],
                                  output_size=10, weight_init_std=weight_init_std)
    
    # Optimize with SGD method
    optimizer = SGD(lr=learning_rate)

    # List for storing the result of accuracy for each networks
    train_acc_list = []
    bn_train_acc_list = []
    
    # Calculates the number of iterations per epoch
    # (= the number of batches that need to be processed in on e epoch)
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    # Iteration with large number
    # but, actually, it will be break down early when the desired number of epoch is reached
    for i in range(1000000000):

        # Random Sampling
        # Randomly selects a batch of data by generating indices to sample from the training data
        batch_mask = np.random.choice(train_size, batch_size)

        # Use the mask to select a batch of input data
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        for _network in (bn_network, network):
            # Compute the graident (with respect to the loss on the batch)
            grads = _network.gradient(x_batch, t_batch)

            # Optimize by using the SGD method
            # (ie. update the parameter)
            optimizer.update(_network.params, grads)
    
        if i % iter_per_epoch == 0:
            # Save the accuracy during each epoch
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)

            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            # print out the accuracy of both networks
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1

            # breaks the loop if the epochs reaches the max
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list, bn_train_acc_list


# Draw the histograms
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print( "============== " + str(i+1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)
    
    plt.subplot(4,4,i+1)
    plt.title("W:" + str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')
    
plt.show()
