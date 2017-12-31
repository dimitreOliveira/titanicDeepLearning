import math
import csv
import numpy as np
import pandas as pd


def load_data(train_path, test_path):
    """
    method for data loading
    :param train_path: path for the train set file
    :param test_path: path for the test set file
    :return: a 'pandas' array for each set
    """

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print("number of training examples = " + str(train_data.shape[0]))
    print("number of test examples = " + str(test_data.shape[0]))
    print("train shape: " + str(train_data.shape))
    print("test shape: " + str(test_data.shape))

    return train_data, test_data


def generate_train_subsets(train_data):
    """
    :param train_data: total train data
    :return: train_dataset 80% of data as train set
             validate_dataset 20% data as validate set
    """
    data_size = train_data.shape[0]
    train_size = math.ceil(data_size * 0.8)
    validate_size = data_size - train_size
    train_dataset = train_data[validate_size:]
    validate_dataset = train_data[:validate_size]

    print('train', train_dataset.shape)
    print('validate', validate_dataset.shape)

    return train_dataset, validate_dataset


def convert_to_one_hot(dataset_size, raw_labels, classes):
    labels = np.zeros((dataset_size, classes))
    labels[np.arange(dataset_size), raw_labels] = 1

    return labels


def output_submission(test_ids, predictions, id_column, predction_column, file_name):
    """
    :param test_ids: vector with test dataset ids
    :param predictions: vector with test dataset predictions
    :param id_column: name of the id column
    :param predction_column: name of the predction column
    :param file_name: string for the output file name
    :return: output a csv with ids ands predictions
    """

    print('Outputting csv...')
    with open(file_name, 'w') as submission:
        writer = csv.writer(submission)
        writer.writerow([id_column, predction_column])
        for test_id, test_prediction in zip(test_ids, np.argmax(predictions, 1)):
            writer.writerow([test_id, test_prediction])
    print('Output complete')
