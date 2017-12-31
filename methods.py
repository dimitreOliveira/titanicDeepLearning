import tensorflow as tf
import numpy as np


def create_placeholders(input_size, output_size):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    input_size -- scalar, input size
    output_size -- scalar, output size

    Returns:
    X -- placeholder for the data input, of shape [None, input_size] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, output_size] and dtype "float"
    """

    X = tf.placeholder(shape=(None, input_size), dtype=tf.float32, name="X")
    Y = tf.placeholder(shape=(None, output_size), dtype=tf.float32, name="Y")

    return X, Y


def initialize_parameters(input_size, output_size, n_nodes):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        w1 : [4, 25]
                        b1 : [25]
                        w2 : [25, 2]
                        b2 : [2]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    w1 = tf.get_variable("w1", [input_size, n_nodes], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [n_nodes], initializer=tf.zeros_initializer())
    w2 = tf.get_variable("w2", [n_nodes, output_size], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [output_size], initializer=tf.zeros_initializer())

    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}

    return parameters


def forward_propagation(x, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    # Z1 = tf.matmul(W1, X) + b1
    # A1 = tf.nn.relu(Z1)
    # Z2 = tf.matmul(W2, A1) + b2

    z1 = tf.matmul(x, w1) + b1
    a1 = tf.nn.relu(z1)
    z2 = tf.matmul(a1, w2) + b2

    return z2


def compute_cost(z3, y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z3, labels=y))

    return cost


def predict(data, parameters):
    tf_test_dataset = tf.cast(tf.constant(data), tf.float32)
    z3_valid = forward_propagation(tf_test_dataset, parameters)
    valid_prediction = tf.nn.softmax(z3_valid)
    prediction = valid_prediction.eval()

    return prediction


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
