import math
import csv
import re
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


def generate_train_subsets(train_data, percentage, print_info=False):
    """
    :param train_data: total train data
    :param percentage: percentage of training set used for actual training
    :param print_info: is True prints information about the derived subsets
    :return: train_dataset 80% of data as train set
             validate_dataset 20% data as validate set
    """
    data_size = train_data.shape[0]
    train_size = math.ceil(data_size * percentage)
    validate_size = data_size - train_size
    train_dataset = train_data[validate_size:]
    validate_dataset = train_data[:validate_size]

    if print_info is True:
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
    :param id_column: name of the output id column
    :param predction_column: name of the output predction column
    :param file_name: string for the output file name
    :return: output a csv with ids ands predictions
    """

    print('Outputting submission...')
    with open('submissions/' + file_name, 'w') as submission:
        writer = csv.writer(submission)
        writer.writerow([id_column, predction_column])
        for test_id, test_prediction in zip(test_ids, np.argmax(predictions, 1)):
            writer.writerow([test_id, test_prediction])
    print('Output complete')


def replace_na_with_mode(dataset, column_name):
    dataset.loc[dataset.Embarked.isnull(), column_name] = dataset[column_name].mode()[0]


def replace_na_with_median(dataset, column_name):
    dataset.loc[dataset.Embarked.isnull(), column_name] = dataset[column_name].median()


def pre_process_data(df):
    # setting `passengerID` as Index since it wont be necessary for the ana lysis
    df = df.set_index("PassengerId")

    # convert 'Sex' values
    df['gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # We see that 2 passengers embarked data is missing, we fill those in as the most common Embarked value
    replace_na_with_mode(df, 'Embarked')

    # convert 'Embarked' values
    df['embarkation'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

    # get titles from the name
    df['title'] = df.apply(lambda row: re.split('[,.]+', row['Name'])[1], axis=1)

    # convert titles to values
    df['title'] = df['title'].map({' Capt': 'Other', ' Master': 'Master', ' Mr': 'Mr', ' Don': 'Other',
                                   ' Dona': 'Other', ' Lady': 'Other', ' Col': 'Other', ' Miss': 'Miss',
                                   ' the Countess': 'Other', ' Dr': 'Other', ' Jonkheer': 'Other', ' Mlle': 'Other',
                                   ' Sir': 'Other', ' Rev': 'Other', ' Ms': 'Other', ' Mme': 'Other', ' Major': 'Other',
                                   ' Mrs': 'Mrs'})
    df = pd.get_dummies(df, columns=['title'])

    # create a new column 'family' as a sum of 'SibSp' and 'Parch'
    # df['family'] = df['SibSp'] + df['Parch']
    # df['family'] = df['family'].map(lambda x: 4 if x > 4 else x)

    # create a new column 'FTicket' as the first character of the 'Ticket'
    df['FTicket'] = df['Ticket'].map(lambda x: x[0])
    df = pd.get_dummies(df, columns=['FTicket'])

    # bin Fare into five intervals with equal amount of values
    df['Fare-bin'] = pd.qcut(df['Fare'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Replace missing age values with median ages by gender
    for gender in df['gender'].unique():
        median_age = df[(df['gender'] == gender)].Age.median()
        df.loc[(df['Age'].isnull()) & (df['gender'] == gender), 'Age'] = median_age

    # bin Age into seven intervals with equal amount of values
    # ('baby','child','teenager','young','mid-age','over-50','senior')
    bins = [0, 4, 12, 18, 30, 50, 65, 100]
    age_index = (1, 2, 3, 4, 5, 6, 7)
    df['Age-bin'] = pd.cut(df['Age'], bins, labels=age_index).astype(int)

    return df
