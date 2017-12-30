import tensorflow as tf


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar
    n_y -- scalar

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    """

    X = tf.placeholder(shape=(None, n_x), dtype=tf.float32, name="X")
    Y = tf.placeholder(shape=(None, n_y), dtype=tf.float32, name="Y")

    return X, Y


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 713]
                        b1 : [25, 1]
                        W2 : [713, 713]
                        b2 : [713, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    W1 = tf.get_variable("W1", [4, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [25, 2], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [2], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
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
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Z1 = tf.matmul(W1, X) + b1
    # A1 = tf.nn.relu(Z1)
    # Z2 = tf.matmul(W2, A1) + b2

    Z1 = tf.matmul(X, W1) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(A1, W2) + b2

    return Z2


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost


def predict(input, parameters):
    tf_test_dataset = tf.cast(tf.constant(input), tf.float32)
    # test_lay_1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    # test_prediction = tf.nn.softmax(tf.matmul(test_lay_1, weights2) + biases2)
    Z3_valid = forward_propagation(tf_test_dataset, parameters)
    valid_prediction = tf.nn.softmax(Z3_valid)
    prediction = valid_prediction.eval()

    # X, _ = create_placeholders(n_x, n_y)
    # prediction = tf.nn.softmax(forward_propagation(input, parameters))
    return prediction
