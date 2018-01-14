import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from methods import compute_cost, create_placeholders, forward_propagation, initialize_parameters, accuracy, \
    l2_regularizer, forward_propagation_dropout
from dataset import generate_train_subsets, mini_batches


def model(train, labels, layers_dims, learning_rate=0.01, num_epochs=15001, train_size=0.8,
          print_cost=True, print_accuracy=True, plot_cost=True, plot_accuracy=False, use_l2=False, l2_beta=0.01,
          use_dropout=False, keep_prob=0.7, hidden_activation='relu', return_max_acc=False, minibatch_size=0,
          lr_decay=0):
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
    :param plot_accuracy: True to plot the train and validation accuracy
    :param use_l2: True to use l2 regularization
    :param l2_beta: beta parameter for the l2 regularization
    :param use_dropout: True to use dropout regularization
    :param keep_prob: probability to keep each node of each hidden layer (dropout)
    :param hidden_activation: activation function to be used on the hidden layers
    :param return_max_acc: True to return the highest accuracy from all epochs
    :param minibatch_size: size of th mini batch
    :param lr_decay: if != 0, sets de learning rate decay on each epoch
    :return parameters: parameters learnt by the model. They can then be used to predict.
    :return submission_name: name for the trained model
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

    train_set, validation_set = generate_train_subsets(train, train_size)
    train_labels, validation_labels = generate_train_subsets(labels, train_size)

    input_size = train_set.shape[1]
    output_size = train_labels.shape[1]
    num_examples = train_set.shape[0]
    n_layers = len(layers_dims)
    train_costs = []
    validation_costs = []
    train_accuracies = []
    validation_accuracies = []
    prediction = []
    best_iteration = [float('inf'), 0, float('-inf'), 0]
    best_acc_params = None
    if minibatch_size == 0:
        minibatch_size = num_examples
    num_minibatches = int(num_examples / minibatch_size)

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
    validation_cost = compute_cost(fw_output_valid, validation_labels)

    if use_l2 is True:
        train_cost = l2_regularizer(train_cost, l2_beta, parameters, n_layers)
        validation_cost = l2_regularizer(validation_cost, l2_beta, parameters, n_layers)

    if lr_decay != 0:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.inverse_time_decay(learning_rate, global_step=global_step, decay_rate=lr_decay,
                                                    decay_steps=1)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost, global_step=global_step)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            train_epoch_cost = 0.
            validation_epoch_cost = 0.

            minibatches = mini_batches(train_set, train_labels, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                feed_dict = {x: minibatch_X, y: minibatch_Y}
                _, minibatch_train_cost, prediction, minibatch_validation_cost = sess.run(
                    [optimizer, train_cost, train_prediction, validation_cost], feed_dict=feed_dict)

                train_epoch_cost += minibatch_train_cost / num_minibatches
                validation_epoch_cost += minibatch_validation_cost / num_minibatches

            validation_accuracy = accuracy(valid_prediction.eval(), validation_labels)
            train_accuracy = accuracy(prediction, minibatch_Y)

            if validation_epoch_cost < best_iteration[0]:
                best_iteration[0] = validation_epoch_cost
                best_iteration[1] = epoch

            if validation_accuracy > best_iteration[2]:
                best_iteration[2] = validation_accuracy
                best_iteration[3] = epoch
                if return_max_acc is True:
                    best_acc_params = sess.run(parameters)

            if print_cost is True and epoch % 500 == 0:
                print("Train cost after epoch %i: %f" % (epoch, train_epoch_cost))
                print("Validation cost after epoch %i: %f" % (epoch, validation_epoch_cost))

            if print_accuracy is True and epoch % 500 == 0:
                print('Train accuracy after epoch {}: {:.2f}'.format(epoch, train_accuracy))
                print('Validation accuracy after epoch {}: {:.2f}'.format(epoch, validation_accuracy))

            if plot_cost is True and epoch % 10 == 0:
                train_costs.append(train_epoch_cost)
                validation_costs.append(validation_epoch_cost)

            if plot_accuracy is True and epoch % 10 == 0:
                train_accuracies.append(train_accuracy)
                validation_accuracies.append(validation_accuracy)

        if return_max_acc is True:
            parameters = best_acc_params
        else:
            parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate accuracy on the train and validation set
        # train_accuracy = accuracy(prediction, train_labels)
        train_accuracy = accuracy(prediction, minibatch_Y)
        validation_accuracy = accuracy(valid_prediction.eval(), validation_labels)
        print('Train accuracy: {:.2f}'.format(train_accuracy))
        print('Validation accuracy: {:.2f}'.format(validation_accuracy))

        # print lowest validation cost from all epochs
        print('Lowest validation cost: {:.2f} at epoch {}'.format(best_iteration[0], best_iteration[1]))

        # print highest validation accuracy from all epochs
        print('Highest validation accuracy: {:.2f} at epoch {}'.format(best_iteration[2], best_iteration[3]))

        # output submission file name
        submission_name = 'tr_acc-{:.2f}-vd_acc{:.2f}-size{}-ly{}-epoch{}.csv'\
            .format(train_accuracy, validation_accuracy, train_size, layers_dims, num_epochs)

        if lr_decay != 0:
            submission_name = 'lrdc{}-'.format(lr_decay) + submission_name
        else:
            submission_name = 'lr{}-'.format(learning_rate) + submission_name

        if use_l2 is True:
            submission_name = 'l2{}-'.format(l2_beta) + submission_name

        if use_dropout is True:
            submission_name = 'dk{}-'.format(keep_prob) + submission_name

        if minibatch_size != num_examples:
            submission_name = 'mb{}-'.format(minibatch_size) + submission_name

        if plot_cost is True:
            plt.plot(np.squeeze(train_costs), label='Train cost')
            plt.plot(np.squeeze(validation_costs), label='Validation cost')
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Model: " + submission_name)
            plt.legend()
            plt.show()

        if plot_accuracy is True:
            plt.plot(np.squeeze(train_accuracies), label='Train accuracy')
            plt.plot(np.squeeze(validation_accuracies), label='Validation accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('iterations (per tens)')
            plt.title("Model: " + submission_name)
            plt.legend()
            plt.show()

        return parameters, submission_name
