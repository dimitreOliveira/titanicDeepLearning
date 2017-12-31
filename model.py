import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from methods import compute_cost, create_placeholders, forward_propagation, initialize_parameters, predict, accuracy
from dataset import load_data, convert_to_one_hot, output_submission, generate_train_subsets


TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'

train, test = load_data(TRAIN_PATH, TEST_PATH)

CLASSES = 2
train_dataset_size = train.shape[0]
train_raw_labels = train.Survived.values

# pre-process
train_pre = train.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'Age', 'Sex'], axis=1).values
test_pre = test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'Age', 'Sex'], axis=1).values

# The labels need to be one-hot encoded
train_labels = convert_to_one_hot(train_dataset_size, train_raw_labels, CLASSES)

X_train, X_test = generate_train_subsets(train_pre)
Y_train, Y_test = generate_train_subsets(train_labels)


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.01,
          num_epochs=3001, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set
    Y_train -- training set labels
    X_test -- validate set
    Y_test -- validate set labels
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    input_size = X_train.shape[1]
    output_size = Y_train.shape[1]
    n_nodes = 25
    costs = []

    x, y = create_placeholders(input_size, output_size)
    tf_valid_dataset = tf.cast(tf.constant(X_test), tf.float32)
    parameters = initialize_parameters(input_size, output_size, n_nodes)

    Z3 = forward_propagation(x, parameters)
    cost = compute_cost(Z3, y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    train_prediction = tf.nn.softmax(Z3)

    Z3_valid = forward_propagation(tf_valid_dataset, parameters)
    valid_prediction = tf.nn.softmax(Z3_valid)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0.
            _, batch_cost, prediction = sess.run([optimizer, cost, train_prediction], feed_dict={x: X_train, y: Y_train})
            epoch_cost += batch_cost

            if print_cost is True and epoch % 500 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost is True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
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
        # print("Train Accuracy:", accuracy.eval({x: X_train, y: Y_train}))
        # print("Test Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
        train_accuracy = accuracy(prediction, Y_train)
        validation_accuracy = accuracy(valid_prediction.eval(), Y_test)
        print('Train accuracy: {:.2f}'.format(train_accuracy))
        print('Validation accuracy: {:.2f}'.format(validation_accuracy))
        # print('Train accuracy: {:.2f}'.format(accuracy(train_prediction.eval(), Y_train)))

        # prediction
        final_prediction = predict(test_pre, parameters)

        # output submission
        submission_name = 'submisson-in{}-epoch{}-lr{}-tr_acc-{:.2f}-vd_acc{:.2f}.csv'\
            .format(input_size, num_epochs, learning_rate, train_accuracy, validation_accuracy)
        output_submission(test.PassengerId.values, final_prediction, 'PassengerId', 'Survived', submission_name)

        return parameters


parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=20001, learning_rate=0.0001)
