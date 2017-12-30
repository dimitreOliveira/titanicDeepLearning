import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from methods import compute_cost, create_placeholders, forward_propagation, initialize_parameters, predict
from dataset import load_data, convert_to_one_hot, output_submission, generate_train_subsets


TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'

train, test = load_data(TRAIN_PATH, TEST_PATH)

CLASSES = 2
train_dataset_size = train.shape[0]
train_raw_labels = train.Survived.values

# pre-process
train_pre = train.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'Age', 'Sex'], axis=1).values
# test_pre = test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'Age', 'Sex'], axis=1).values

# The labels need to be one-hot encoded
train_labels = convert_to_one_hot(train_dataset_size, train_raw_labels, CLASSES)

X_train, X_test = generate_train_subsets(train_pre)
Y_train, Y_test = generate_train_subsets(train_labels)


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.01,
          num_epochs=301, print_cost=True):
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
    n_x = X_train.shape[1]  # n_y : input size
    n_y = Y_train.shape[1]  # n_y : output size
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    tf_valid_dataset = tf.cast(tf.constant(X_test), tf.float32)
    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    train_prediction = tf.nn.softmax(Z3)

    Z3_valid = forward_propagation(tf_valid_dataset, parameters)
    valid_prediction = tf.nn.softmax(Z3_valid)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0.
            # _, batch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            _, batch_cost, prediction = sess.run([optimizer, cost, train_prediction], feed_dict={X: X_train, Y: Y_train})
            epoch_cost += batch_cost

            if print_cost is True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost is True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        print('pred2')
        print(np.sum(np.argmax(prediction, 1) == np.argmax(Y_train, 1)) / prediction.shape[0])
        # print(np.sum(np.argmax(train_prediction.eval(), 1) == np.argmax(Y_train, 1)) / train_prediction.eval().shape[0])
        print(np.sum(np.argmax(valid_prediction.eval(), 1) == np.argmax(Y_test, 1)) / valid_prediction.eval().shape[0])

        print('predictions')
        final_prediction = predict(X_test, parameters)
        # print(final_prediction)

        output_submission(test.PassengerId.values, final_prediction, 'PassengerId', 'Survived',
                          'titanic_submission2.csv')

        return parameters


parameters = model(X_train, Y_train, X_test, Y_test)



# output_submission(test.PassengerId.values, final_prediction, 'PassengerId', 'Survived', 'titanic_submission.csv')