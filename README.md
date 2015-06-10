# Vectorized-Neural-Network

Python implementation of Neural Network class, fully vectorized and makes use on numpy's matrix multiplication for  better performance.

Can be used to create a neural net with as many hidden layers as needed.

Usage:

    # create a network with 10 input, 5 hidden, and 2 output nodes
    n = neuralNet(10, 5, 2)
    # training
    n.learn(data_train)
    #test
    n.test(data_test)
