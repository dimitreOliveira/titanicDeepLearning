import pandas as pd
import numpy as np
from model import model
from methods import predict
from dataset import load_data, convert_to_one_hot, pre_process_data, output_submission


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


train = pre_process_data(train)
test = pre_process_data(test)


# drop unwanted columns
train_pre = train.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'Sex', 'Fare', 'Age'], axis=1).values
test_pre = test.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Fare', 'Age'], axis=1).values


# The labels need to be one-hot encoded
train_labels = convert_to_one_hot(train_dataset_size, train_raw_labels, CLASSES)

input_layer = train_pre.shape[1]
output_layer = 2
num_epochs = 5001
learning_rate = 0.01
train_size = 0.8
layers_dims = [input_layer, 20, 10, output_layer]

trained_parameters, submission_name = model(train_pre, train_labels, layers_dims, train_size=train_size,
                                            num_epochs=num_epochs, learning_rate=learning_rate, use_l2=True,
                                            use_dropout=True, print_cost=False, print_accuracy=False, plot_cost=True,
                                            plot_accuracy=True, l2_beta=0.01, keep_prob=0.8, return_max_acc=True)
print(submission_name)
final_prediction = predict(test_pre, trained_parameters)
output_submission(test.index.values, final_prediction, 'PassengerId', 'Survived', submission_name)
