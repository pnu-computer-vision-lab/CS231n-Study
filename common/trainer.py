# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.optimizer import *

class Trainer:
    """
    Class which encapsulates the process of training a neural network
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # Optimizer Dictionary
        # maps string keys to optimizer classes and then creates an instance of the optimizer using the provided parameters.
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)

        # Calculates the max number of iterations for training
        self.max_iter = int(epochs * self.iter_per_epoch)

        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        """
        Executes a single step of training
        """

        # Randomly selects a batch of indices from the training set
        batch_mask = np.random.choice(self.train_size, self.batch_size)

        # Select a subset of the training data
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        # Calculates the gradient of the network's loss with respect to its parameters on its batch
        grads = self.network.gradient(x_batch, t_batch)

        # Update the network's parameters based on the gradients
        self.optimizer.update(self.network.params, grads)
        
        # Calculate and records the training loss
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        if self.verbose: print("train loss:" + str(loss))
        
        # Checks if an epoch has completed
        # If an epoch is completed, evaluates the network's performance on both the training and test sets.
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test

            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        """
        Performs the entire training process
        """

        # Runs the training for the calculated number of iterations
        for i in range(self.max_iter):
            self.train_step()
        
        # Evaluate the final accuracy on the test set
        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

