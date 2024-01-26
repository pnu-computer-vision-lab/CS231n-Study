# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

"""
    Start from randomly initialized two hyperparameters (leraning rate, weight decay)
    Optimize these hyperparameters
"""

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Truncate the trining data so as to get faster results
x_train = x_train[:500]
t_train = t_train[:500]

"""
    Train data: Train the parameters
    Validation data: Estimate the performance of the hyperparameters
    Test data: Estimate the common performance of the neural network
"""

# Divide 20% into validation set
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)

# Shuffle the data set
x_train, t_train = shuffle_dataset(x_train, t_train)

# Divide the data set into validation set and training set

# Validation set: dataset of the range [0:val_num]
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]

# Training set: dataset of the range [val_num:]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

def __train(lr, weight_decay, epocs=50):
    """
    Perform the training stage

    Params:
        lr: learning rate
        weight_decay: the strength of the weight_decay
        epocs: the number of epochs
    Return:
        trainer.test_acc_list: a list of the accuracy with testing stage
        trainer.train_acc_list: a list of the accuracy with training stage
    """

    # Instance of the network
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    
    # Instance of the neural network training

    # mini batch size is 100 / optimizer method is 'sgd'
    # the number of epochs is 50
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    

    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# Random exploration of the hyperparameters

# Indicates the number of trial for optimization
optimization_trial = 100

# Dictionary for saving the results of validation and train set
results_val = {}
results_train = {}

for _ in range(optimization_trial):

    # Range of the hyperparameters: weight decay lambda & learning rate
    # Each of these hyperparameters are selected by random selection
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)

    val_acc_list, train_acc_list = __train(lr, weight_decay)

    # Print the result validation accuracy according to each learning rate and weight decay
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))

    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# Draw the histograms
    
# Sort and Display histograms of the results in descending order
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()
