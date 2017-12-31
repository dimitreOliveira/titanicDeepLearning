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

    x = tf.placeholder(shape=(None, input_size), dtype=tf.float32, name="X")
    y = tf.placeholder(shape=(None, output_size), dtype=tf.float32, name="Y")

    return x, y


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


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    print(L)

    # Implement [LINEAR -> RELU]*(L-1).
    for l in range(1, L):
        print(l)
        A_prev = A
        A = linear_activation_forward(A_prev, parameters['w%s' % l], parameters['b%s' % l], 'relu')

    # Last layer must output only LINEAR
    AL = tf.matmul(A, parameters['w%s' % L]) + parameters['b%s' % L]

    return AL


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z = tf.matmul(A_prev, W) + b
        A = tf.nn.sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z = tf.matmul(A_prev, W) + b
        A = tf.nn.relu(Z)

    return A


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        # parameters['w' + str(l)] = np.dot(np.random.randn(layer_dims[l], layer_dims[l - 1]), 0.01)
        # parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        parameters['w' + str(l)] = tf.get_variable('w' + str(l), [layer_dims[l - 1], layer_dims[l]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters['b' + str(l)] = tf.get_variable('b' + str(l), [layer_dims[l]], initializer=tf.zeros_initializer())

    return parameters


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
    """
    make a prediction based on a data set and parameters
    :param data: based data set
    :param parameters: based parameters
    :return: array of predictions
    """
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        dataset = tf.cast(tf.constant(data), tf.float32)
        fw_prop_result = forward_propagation(dataset, parameters)
        fw_prop_activation = tf.nn.softmax(fw_prop_result)
        prediction = fw_prop_activation.eval()

    return prediction


def accuracy(predictions, labels):
    """
    calculate accuracy between two data sets
    :param predictions: data set of predictions
    :param labels: data set of labels (real values)
    :return: percentage of correct predictions
    """
    prediction_size = predictions.shape[0]
    prediction_accuracy = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / prediction_size
    return 100.0 * prediction_accuracy
