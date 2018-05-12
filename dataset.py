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


def pre_process_data(df):
    """
    Perform a number of pre process functions on the data set
    :param df: pandas data frame
    :return: updated data frame
    """
    # setting `passengerID` as Index since it wont be necessary for the analysis
    df = df.set_index("PassengerId")

    # convert 'Sex' values
    df['gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # We see that 2 passengers embarked data is missing, we fill those in as the most common Embarked value
    df.loc[df.Embarked.isnull(), 'Embarked'] = df['Embarked'].mode()[0]

    # Replace missing age values with median ages by gender
    for gender in df['gender'].unique():
        median_age = df[(df['gender'] == gender)].Age.median()
        df.loc[(df['Age'].isnull()) & (df['gender'] == gender), 'Age'] = median_age

    # convert 'gender' values to new columns
    df = pd.get_dummies(df, columns=['gender'])

    # convert 'Embarked' values to new columns
    df = pd.get_dummies(df, columns=['Embarked'])

    # bin Fare into five intervals with equal amount of values
    df['Fare-bin'] = pd.qcut(df['Fare'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # bin Age into seven intervals with equal amount of values
    # ('baby','child','teenager','young','mid-age','over-50','senior')
    bins = [0, 4, 12, 18, 30, 50, 65, 100]
    age_index = (1, 2, 3, 4, 5, 6, 7)
    df['Age-bin'] = pd.cut(df['Age'], bins, labels=age_index).astype(int)

    # create a new column 'family' as a sum of 'SibSp' and 'Parch'
    df['family'] = df['SibSp'] + df['Parch'] + 1
    df['family'] = df['family'].map(lambda x: 4 if x > 4 else x)

    # create a new column 'FTicket' as the first character of the 'Ticket'
    df['FTicket'] = df['Ticket'].map(lambda x: x[0])
    # combine smaller categories into one
    df['FTicket'] = df['FTicket'].replace(['W', 'F', 'L', '5', '6', '7', '8', '9'], '4')
    # convert 'FTicket' values to new columns
    df = pd.get_dummies(df, columns=['FTicket'])

    # get titles from the name
    df['title'] = df.apply(lambda row: re.split('[,.]+', row['Name'])[1], axis=1)

    # convert titles to values
    df['title'] = df['title'].map({' Capt': 'Other', ' Master': 'Master', ' Mr': 'Mr', ' Don': 'Other',
                                   ' Dona': 'Other', ' Lady': 'Other', ' Col': 'Other', ' Miss': 'Miss',
                                   ' the Countess': 'Other', ' Dr': 'Other', ' Jonkheer': 'Other', ' Mlle': 'Other',
                                   ' Sir': 'Other', ' Rev': 'Other', ' Ms': 'Other', ' Mme': 'Other', ' Major': 'Other',
                                   ' Mrs': 'Mrs'})
    # convert 'title' values to new columns
    df = pd.get_dummies(df, columns=['title'])

    df = df.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Fare', 'Age'], axis=1)

    return df


def mini_batches(train_set, train_labels, mini_batch_size):
    """
    Generate mini batches from the data set (data and labels)
    :param train_set: data set with the examples
    :param train_labels: data set with the labels
    :param mini_batch_size: mini batch size
    :return: mini batches
    """
    set_size = train_set.shape[0]
    batches = []
    num_complete_minibatches = set_size // mini_batch_size

    for k in range(0, num_complete_minibatches):
        mini_batch_x = train_set[k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_y = train_labels[k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_x, mini_batch_y)
        batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if set_size % mini_batch_size != 0:
        mini_batch_x = train_set[(set_size - (set_size % mini_batch_size)):]
        mini_batch_y = train_labels[(set_size - (set_size % mini_batch_size)):]
        mini_batch = (mini_batch_x, mini_batch_y)
        batches.append(mini_batch)

    return batches
