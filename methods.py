import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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


def forward_propagation(x, parameters, hidden_activation='relu'):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR-> computation
    Arguments:
    x -- data, pandas array of shape (input size, number of examples)
    parameters -- output of initialize_parameters()
    Returns:
    al -- last LINEAR value
    """

    a = x
    n_layers = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1).
    for l in range(1, n_layers):
        a_prev = a
        a = linear_activation_forward(a_prev, parameters['w%s' % l], parameters['b%s' % l], hidden_activation)

    # Last layer must output only LINEAR
    al = tf.matmul(a, parameters['w%s' % n_layers]) + parameters['b%s' % n_layers]

    return al


def forward_propagation_dropout(x, parameters, keep_prob, hidden_activation='relu'):
    """
    Implement forward propagation with dropout for the [LINEAR->RELU]*(L-1)->LINEAR-> computation
    Arguments:
    x -- data, pandas array of shape (input size, number of examples)
    parameters -- output of initialize_parameters()
    parameters -- keep_prob = probability to keep each node of the layer
    Returns:
    al -- last LINEAR value
    """

    a_dropout = x
    n_layers = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1), with dropout
    for l in range(1, n_layers):
        a_prev = a_dropout
        a = linear_activation_forward(a_prev, parameters['w%s' % l], parameters['b%s' % l], hidden_activation)
        a_dropout = tf.nn.dropout(a, keep_prob)

    # Last layer must output only LINEAR
    al = tf.matmul(a_dropout, parameters['w%s' % n_layers]) + parameters['b%s' % n_layers]
    # al_dropout = tf.nn.dropout(al, keep_prob)

    return al


def linear_activation_forward(a_prev, w, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    Arguments:
    a_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    w -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    Returns:
    a -- the output of the activation function, also called the post-activation value
    """

    a = None
    if activation == "sigmoid":
        z = tf.matmul(a_prev, w) + b
        a = tf.nn.sigmoid(z)

    elif activation == "relu":
        z = tf.matmul(a_prev, w) + b
        a = tf.nn.relu(z)

    elif activation == "leaky relu":
        z = tf.matmul(a_prev, w) + b
        a = tf.nn.leaky_relu(z)

    return a


def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    Returns:
    parameters -- python dictionary containing your parameters "w1", "b1", ..., "wn", "bn":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}
    n_layers = len(layer_dims)  # number of layers in the network

    for l in range(1, n_layers):
        parameters['w' + str(l)] = tf.get_variable('w' + str(l), [layer_dims[l - 1], layer_dims[l]],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed=1))
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

    return 100 * prediction_accuracy


def minibatch_accuracy(predictions, labels):
    """
    calculate accuracy between two data sets
    :param predictions: data set of predictions
    :param labels: data set of labels (real values)
    :return: percentage of correct predictions
    """

    prediction_size = predictions.shape[0]
    prediction_accuracy = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / prediction_size

    return 100 * prediction_accuracy


def l2_regularizer(cost, l2_beta, parameters, n_layers):
    """
    Function to apply l2 regularization to the model
    :param cost: usual cost of the model
    :param l2_beta: beta value used for the normalization
    :param parameters: parameters from the model (used to get weights values)
    :param n_layers: number of layers of the model
    :return: cost updated
    """

    regularizer = 0
    for i in range(1, n_layers):
        regularizer += tf.nn.l2_loss(parameters['w%s' % i])
        cost = tf.reduce_mean(cost + l2_beta * regularizer)

    return cost


def build_submission_name(train_accuracy, validation_accuracy, train_size, layers_dims, num_epochs, lr_decay,
                          learning_rate, use_l2, l2_beta, use_dropout, keep_prob, minibatch_size, num_examples):
    submission_name = 'tr_acc-{:.2f}-vd_acc{:.2f}-size{}-ly{}-epoch{}.csv' \
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

    return submission_name


def plot_model_cost(train_costs, validation_costs, submission_name):
    plt.plot(np.squeeze(train_costs), label='Train cost')
    plt.plot(np.squeeze(validation_costs), label='Validation cost')
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Model: " + submission_name)
    plt.legend()
    plt.show()


def plot_model_accuracy(train_accuracies, validation_accuracies, submission_name):
    plt.plot(np.squeeze(train_accuracies), label='Train accuracy')
    plt.plot(np.squeeze(validation_accuracies), label='Validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('iterations (per tens)')
    plt.title("Model: " + submission_name)
    plt.legend()
    plt.show()
