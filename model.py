import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
# from methods import compute_cost, create_placeholders, forward_propagation, initialize_parameters, accuracy
from methods import *
from dataset import generate_train_subsets


def model(train, labels, learning_rate=0.01, num_epochs=15001, train_size=0.8,
          print_cost=True, plot_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    train_set -- training set
    train_labels -- training set labels
    validation_set -- validate set
    validation_labels -- validate set labels
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    print_cost -- True to print the cost every 500 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

    # generate train and labels sub sets
    train_set, validation_set = generate_train_subsets(train, train_size)
    train_labels, validation_labels = generate_train_subsets(labels, train_size)

    input_size = train_set.shape[1]
    output_size = train_labels.shape[1]
    n_nodes = 25
    costs = []
    prediction = []

    x, y = create_placeholders(input_size, output_size)
    tf_valid_dataset = tf.cast(tf.constant(validation_set), tf.float32)
    # parameters = initialize_parameters(input_size, output_size, n_nodes)
    parameters = initialize_parameters_deep([input_size, n_nodes, output_size])

    # z3 = forward_propagation(x, parameters)
    z3 = L_model_forward(x, parameters)
    cost = compute_cost(z3, y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    train_prediction = tf.nn.softmax(z3)

    z3_valid = forward_propagation(tf_valid_dataset, parameters)
    valid_prediction = tf.nn.softmax(z3_valid)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0.
            feed_dict = {x: train_set, y: train_labels}
            _, batch_cost, prediction = sess.run([optimizer, cost, train_prediction], feed_dict=feed_dict)
            epoch_cost += batch_cost

            if print_cost is True and epoch % 500 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost is True and epoch % 5 == 0:
                costs.append(epoch_cost)

        if plot_cost is True:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate accuracy on the train and validation set
        train_accuracy = accuracy(prediction, train_labels)
        validation_accuracy = accuracy(valid_prediction.eval(), validation_labels)
        print('Train accuracy: {:.2f}'.format(train_accuracy))
        print('Validation accuracy: {:.2f}'.format(validation_accuracy))

        # output submission file name
        submission_name = 'submisson-tr_acc-{:.2f}-vd_acc{:.2f}-in{}-lr{}-size{}-epoch{}.csv'\
            .format(train_accuracy, validation_accuracy, input_size, learning_rate, train_size, num_epochs)

        return parameters, submission_name
