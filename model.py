import tensorflow as tf
from tensorflow.python.framework import ops
from methods import compute_cost, create_placeholders, forward_propagation, initialize_parameters, accuracy, \
    l2_regularizer, build_submission_name, plot_model_cost, plot_model_accuracy, predict
from dataset import mini_batches


def model(train_set, train_labels, validation_set, validation_labels, layers_dims, learning_rate=0.01, num_epochs=1001,
          print_cost=True, plot_cost=True, l2_beta=0., keep_prob=1.0, hidden_activation='relu', return_best=False,
          minibatch_size=0, lr_decay=0, print_accuracy=True, plot_accuracy=True):
    """
    :param train_set: training set
    :param train_labels: training labels
    :param validation_set: validation set
    :param validation_labels: validation labels
    :param layers_dims: array with the layer for the model
    :param learning_rate: learning rate of the optimization
    :param num_epochs: number of epochs of the optimization loop
    :param print_cost: True to print the cost every 500 epochs
    :param plot_cost: True to plot the train and validation cost
    :param l2_beta: beta parameter for the l2 regularization
    :param keep_prob: probability to keep each node of each hidden layer (dropout)
    :param hidden_activation: activation function to be used on the hidden layers
    :param return_best: True to return the highest params from all epochs
    :param minibatch_size: size of th mini batch
    :param lr_decay: if != 0, sets de learning rate decay on each epoch
    :param print_accuracy: True to print the accuracy every 500 epochs
    :param plot_accuracy: True to plot the train and validation accuracy
    :return parameters: parameters learnt by the model. They can then be used to predict.
    :return submission_name: name for the trained model
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

    input_size = layers_dims[0]
    output_size = layers_dims[-1]
    num_examples = train_set.shape[0]
    n_layers = len(layers_dims)
    train_costs = []
    validation_costs = []
    train_accuracies = []
    validation_accuracies = []
    prediction = []
    best_iteration = [float('inf'), 0, float('-inf'), 0]
    best_params = None

    if minibatch_size == 0 or minibatch_size > num_examples:
        minibatch_size = num_examples

    num_minibatches = num_examples // minibatch_size

    if num_minibatches == 0:
        num_minibatches = 1

    submission_name = build_submission_name(layers_dims, num_epochs, lr_decay, learning_rate, l2_beta, keep_prob,
                                            minibatch_size, num_examples)

    x, y = create_placeholders(input_size, output_size)
    tf_valid_dataset = tf.cast(tf.constant(validation_set), tf.float32)
    parameters = initialize_parameters(layers_dims)

    fw_output = forward_propagation(x, parameters, keep_prob, hidden_activation)
    train_cost = compute_cost(fw_output, y)
    train_prediction = tf.nn.softmax(fw_output)

    fw_output_valid = forward_propagation(tf_valid_dataset, parameters, 1, hidden_activation)
    validation_cost = compute_cost(fw_output_valid, validation_labels)
    valid_prediction = tf.nn.softmax(fw_output_valid)

    if l2_beta > 0:
        train_cost = l2_regularizer(train_cost, l2_beta, parameters, n_layers)
        validation_cost = l2_regularizer(validation_cost, l2_beta, parameters, n_layers)

    if lr_decay != 0:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.inverse_time_decay(learning_rate, global_step=global_step, decay_rate=lr_decay,
                                                    decay_steps=1)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost, global_step=global_step)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost)

    # uncomment to use tensorboard
    # tf.summary.scalar('train cost', train_cost)
    # tf.summary.scalar('validation cost', validation_cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # uncomment to use tensorboard
        # writer = tf.summary.FileWriter('logs/'+submission_name, sess.graph)

        sess.run(init)

        for epoch in range(num_epochs):
            train_epoch_cost = 0.
            validation_epoch_cost = 0.

            minibatches = mini_batches(train_set, train_labels, minibatch_size)

            for minibatch in minibatches:
                # uncomment to use tensorboard
                # merge = tf.summary.merge_all()

                (minibatch_X, minibatch_Y) = minibatch
                feed_dict = {x: minibatch_X, y: minibatch_Y}

                # uncomment to use tensorboard
                # _, summary, minibatch_train_cost, prediction, minibatch_validation_cost = sess.run(
                #     [optimizer, merge, train_cost, train_prediction, validation_cost], feed_dict=feed_dict)

                # comment to use tensorboard
                _, minibatch_train_cost, prediction, minibatch_validation_cost = sess.run(
                    [optimizer, train_cost, train_prediction, validation_cost], feed_dict=feed_dict)

                train_epoch_cost += minibatch_train_cost / num_minibatches
                validation_epoch_cost += minibatch_validation_cost / num_minibatches

            validation_accuracy = accuracy(valid_prediction.eval(), validation_labels)
            train_accuracy = accuracy(prediction, minibatch_Y)

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

            # uncomment to use tensorboard
            # if epoch % 10 == 0:
            #     writer.add_summary(summary, epoch)

            if return_best is True:
                if validation_epoch_cost < best_iteration[0]:
                    best_iteration[0] = validation_epoch_cost
                    best_iteration[1] = epoch
                    best_params = sess.run(parameters)
                if validation_accuracy > best_iteration[2]:
                    best_iteration[2] = validation_accuracy
                    best_iteration[3] = epoch
                    best_params = sess.run(parameters)

        if return_best is True:
            parameters = best_params
        else:
            parameters = sess.run(parameters)

        print("Parameters have been trained, getting metrics...")

        train_accuracy = accuracy(predict(train_set, parameters), train_labels)
        validation_accuracy = accuracy(predict(validation_set, parameters), validation_labels)

        print('Train accuracy: {:.2f}'.format(train_accuracy))
        print('Validation accuracy: {:.2f}'.format(validation_accuracy))
        print('Lowest validation cost: {:.2f} at epoch {}'.format(best_iteration[0], best_iteration[1]))
        print('Highest validation accuracy: {:.2f} at epoch {}'.format(best_iteration[2], best_iteration[3]))

        submission_name = 'tr_acc-{:.2f}-vd_acc{:.2f}-'.format(train_accuracy, validation_accuracy) + submission_name

        if return_best is True:
            print('Lowest cost: {:.2f} at epoch {}'.format(best_iteration[0], best_iteration[1]))

        if plot_cost is True:
            plot_model_cost(train_costs, validation_costs, submission_name)

        if plot_accuracy is True:
            plot_model_accuracy(train_accuracies, validation_accuracies, submission_name)

        return parameters, submission_name
