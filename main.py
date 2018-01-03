import pandas as pd
import numpy as np
import re
from model import model
from methods import predict
from dataset import load_data, convert_to_one_hot, output_submission


# optional, sets output of data when printed
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)


TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'

train, test = load_data(TRAIN_PATH, TEST_PATH)

CLASSES = 2
train_dataset_size = train.shape[0]
train_raw_labels = train.Survived.values

# pre-process
# convert 'Sex' values
train['gender'] = train['Sex'].map({'female': 0, 'male': 1}).astype(int)
test['gender'] = test['Sex'].map({'female': 0, 'male': 1}).astype(int)

# We see that 2 passengers embarked data is missing, we fill those in as the most common port, S
train.loc[train.Embarked.isnull(), 'Embarked'] = 'S'  # verificar esse valor da moda dinamicamente

# convert 'Embarked' values
train['embarkation'] = train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
test['embarkation'] = test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

# get titles from the name
train['title'] = train.apply(lambda row: re.split('[,.]+', row['Name'])[1], axis=1)
test['title'] = test.apply(lambda row: re.split('[,.]+', row['Name'])[1], axis=1)

# convert titles to values
# maps commoner = 0 => Mr, Ms, Miss, Ms
# high-commoner = 1 => Dr, Master, Mme, Mlle
# Officer = 2 => Capt, Col, Major
# Clerc = 3 => Rev
# Noble = 4 => Don, Dona, Sir, Lady
# high-Noble = 5 => the Countess, Jonkheer
train['title'] = train['title'].map({' Capt': 2, ' Master': 1, ' Mr': 0, ' Don': 4, ' Dona': 4, ' Lady': 4, ' Col': 2,
                                     ' Miss': 0, ' the Countess': 5, ' Dr': 1, ' Jonkheer': 5, ' Mlle': 1, ' Sir': 4,
                                     ' Rev': 3, ' Ms': 0, ' Mme': 1, ' Major': 2, ' Mrs': 0}).astype(int)
test['title'] = test['title'].map({' Capt': 2, ' Master': 1, ' Mr': 0, ' Don': 4, ' Dona': 4, ' Lady': 4, ' Col': 2,
                                   ' Miss': 0, ' the Countess': 5, ' Dr': 1, ' Jonkheer': 5, ' Mlle': 1, ' Sir': 4,
                                   ' Rev': 3, ' Ms': 0, ' Mme': 1, ' Major': 2, ' Mrs': 0}).astype(int)


# There are passengers with missing age data, so we need to fill that in
# We want to get median ages by gender
genders = [0, 1]
for gender in genders:
    # We locate every passenger of given gender whose age is null, and assign it to the median age
    median_age = train[(train.gender == gender)].Age.dropna().median()
    train.loc[(train.Age.isnull()) & (train.gender == gender), 'Age'] = median_age


train_pre = train.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'Sex'], axis=1).values
test_pre = test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'Sex'], axis=1).values

# print(train)
# print(train.shape)

# The labels need to be one-hot encoded
train_labels = convert_to_one_hot(train_dataset_size, train_raw_labels, CLASSES)

input_layer = train_pre.shape[1]
output_layer = 2
num_epochs = 5001
learning_rate = 0.001  # 0.01
train_size = 0.7
layers_dims = [input_layer, 50, 40, 30, 20, 10, 5, output_layer]


trained_parameters, submission_name = model(train_pre, train_labels, layers_dims, train_size=train_size,
                                            num_epochs=num_epochs, learning_rate=learning_rate, print_cost=False, plot_cost=False)
print(submission_name)


final_prediction = predict(test_pre, trained_parameters)
output_submission(test.PassengerId.values, final_prediction, 'PassengerId', 'Survived', submission_name)
