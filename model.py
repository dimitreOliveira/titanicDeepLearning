import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from methods import compute_cost, create_placeholders, forward_propagation, initialize_parameters, predict, accuracy
from dataset import output_submission


def model(train_set, train_labels, validation_set, validation_labels, test_set, pre_test_set, learning_rate=0.01,
          num_epochs=3001, print_cost=True, plot_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    train_set -- training set
    train_labels -- training set labels
    validation_set -- validate set
    validation_labels -- validate set labels
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    input_size = train_set.shape[1]
    output_size = train_labels.shape[1]
    n_nodes = 25
    costs = []
    prediction = []

    x, y = create_placeholders(input_size, output_size)
    tf_valid_dataset = tf.cast(tf.constant(validation_set), tf.float32)
    parameters = initialize_parameters(input_size, output_size, n_nodes)

    z3 = forward_propagation(x, parameters)
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

        # plot the cost
        if plot_cost is True:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        # correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(y))

        # Calculate accuracy on the test set
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # print("Train Accuracy:", accuracy.eval({x: train_set, y: train_labels}))
        # print("Test Accuracy:", accuracy.eval({x: validation_set, y: validation_labels}))
        train_accuracy = accuracy(prediction, train_labels)
        validation_accuracy = accuracy(valid_prediction.eval(), validation_labels)
        print('Train accuracy: {:.2f}'.format(train_accuracy))
        print('Validation accuracy: {:.2f}'.format(validation_accuracy))
        # print('Train accuracy: {:.2f}'.format(accuracy(train_prediction.eval(), train_labels)))

        # prediction
        final_prediction = predict(pre_test_set, parameters)

        # output submission
        submission_name = 'submisson-in{}-epoch{}-lr{}-tr_acc-{:.2f}-vd_acc{:.2f}.csv'\
            .format(input_size, num_epochs, learning_rate, train_accuracy, validation_accuracy)
        output_submission(test_set.PassengerId.values, final_prediction, 'PassengerId', 'Survived', submission_name)

        return parameters