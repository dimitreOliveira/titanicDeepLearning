import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from methods import compute_cost, create_placeholders, forward_propagation, initialize_parameters, accuracy, l2_regularizer, forward_propagation_dropout
from dataset import generate_train_subsets


def model(train, labels, layers_dims, learning_rate=0.01, num_epochs=15001, train_size=0.8,
          print_cost=True, print_accuracy=True, plot_cost=True, use_l2=False, l2_beta=0.01, use_dropout=False, keep_prob=0.7, hidden_activation='relu'):
    """
    Implements a n-layer tensorflow neural network: LINEAR->RELU*(n times)->LINEAR->SOFTMAX.
    :param train: training set
    :param labels: validation set
    :param layers_dims: array with the layer for the model
    :param learning_rate: learning rate of the optimization
    :param num_epochs: number of epochs of the optimization loop
    :param train_size: percentage of the train set to use on training
    :param print_cost: True to print the cost every 500 epochs
    :param print_accuracy: True to print the accuracy every 500 epochs
    :param plot_cost: True to plot the train and validation cost
    :param use_l2: True to use l2 regularization
    :param l2_beta: beta parameter for the l2 regularization
    :param use_dropout: True to use dropout regularization
    :param keep_prob: probability to keep each node of each hidden layer (dropout)
    :param hidden_activation: activation function to be used on the hidden layers
    :return parameters: parameters learnt by the model. They can then be used to predict.
    :return submission_name: name for the trained model
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

    # generate train and labels sub sets
    train_set, validation_set = generate_train_subsets(train, train_size)
    train_labels, validation_labels = generate_train_subsets(labels, train_size)

    input_size = train_set.shape[1]
    output_size = train_labels.shape[1]
    n_layers = len(layers_dims)
    train_costs = []
    test_costs = []
    prediction = []
    best_iteration = [float('inf'), 0, float('-inf'), 0]

    x, y = create_placeholders(input_size, output_size)
    tf_valid_dataset = tf.cast(tf.constant(validation_set), tf.float32)
    parameters = initialize_parameters(layers_dims)

    if use_dropout is True:
        fw_output = forward_propagation_dropout(x, parameters, keep_prob, hidden_activation)
    else:
        fw_output = forward_propagation(x, parameters, hidden_activation)
    train_prediction = tf.nn.softmax(fw_output)
    train_cost = compute_cost(fw_output, y)

    fw_output_valid = forward_propagation(tf_valid_dataset, parameters, hidden_activation)
    valid_prediction = tf.nn.softmax(fw_output_valid)
    test_cost = compute_cost(fw_output_valid, validation_labels)

    if use_l2 is True:
        train_cost = l2_regularizer(train_cost, l2_beta, parameters, n_layers)
        test_cost = l2_regularizer(test_cost, l2_beta, parameters, n_layers)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            train_epoch_cost = 0.
            test_epoch_cost = 0.
            feed_dict = {x: train_set, y: train_labels}
            _, train_batch_cost, prediction, test_batch_cost = sess.run([optimizer, train_cost, train_prediction,
                                                                         test_cost], feed_dict=feed_dict)
            train_epoch_cost += train_batch_cost
            test_epoch_cost += test_batch_cost

            validation_accuracy = accuracy(valid_prediction.eval(), validation_labels)

            if test_epoch_cost < best_iteration[0]:
                best_iteration[0] = test_epoch_cost
                best_iteration[1] = epoch

            if validation_accuracy > best_iteration[2]:
                best_iteration[2] = validation_accuracy
                best_iteration[3] = epoch

            if print_cost is True and epoch % 500 == 0:
                print("Train cost after epoch %i: %f" % (epoch, train_epoch_cost))
                print("Validation cost after epoch %i: %f" % (epoch, test_epoch_cost))

            if print_accuracy is True and epoch % 500 == 0:
                train_accuracy = accuracy(prediction, train_labels)
                validation_accuracy = accuracy(valid_prediction.eval(), validation_labels)
                print('Train accuracy after epoch {}: {:.2f}'.format(epoch, train_accuracy))
                print('Validation accuracy after epoch {}: {:.2f}'.format(epoch, validation_accuracy))
            if plot_cost is True and epoch % 10 == 0:
                train_costs.append(train_epoch_cost)
                test_costs.append(test_epoch_cost)

        if plot_cost is True:
            plt.plot(np.squeeze(train_costs), label='Train cost')
            plt.plot(np.squeeze(test_costs), label='Validation cost')
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.legend()
            plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate accuracy on the train and validation set
        train_accuracy = accuracy(prediction, train_labels)
        validation_accuracy = accuracy(valid_prediction.eval(), validation_labels)
        print('Train accuracy: {:.2f}'.format(train_accuracy))
        print('Validation accuracy: {:.2f}'.format(validation_accuracy))

        # print lowest test cost from all epochs
        print('Lowest validation cost: {:.2f} at epoch {}'.format(best_iteration[0], best_iteration[1]))

        # print highest test accuracy from all epochs
        print('Highest validation accuracy: {:.2f} at epoch {}'.format(best_iteration[2], best_iteration[3]))

        # output submission file name
        submission_name = 'tr_acc-{:.2f}-vd_acc{:.2f}-lr{}-size{}-ly{}-epoch{}.csv'\
            .format(train_accuracy, validation_accuracy, learning_rate, train_size, layers_dims, num_epochs)

        if use_l2 is True:
            submission_name = 'l2{}-'.format(l2_beta) + submission_name

        if use_dropout is True:
            submission_name = 'dk{}-'.format(keep_prob) + submission_name
        return parameters, submission_name
