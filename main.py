import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from model import model
from methods import predict
from dataset import load_data, pre_process_data, output_submission


TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'

train, test = load_data(TRAIN_PATH, TEST_PATH)

CLASSES = 2
train_dataset_size = train.shape[0]
# The labels need to be one-hot encoded
train_raw_labels = pd.get_dummies(train.Survived).as_matrix()

train = pre_process_data(train)
test = pre_process_data(test)

# drop unwanted columns
train_pre = train.drop(['Survived'], axis=1).as_matrix().astype(np.float)
test_pre = test.as_matrix().astype(np.float)


# scale values
standard_scaler = preprocessing.StandardScaler()
train_pre = standard_scaler.fit_transform(train_pre)
test_pre = standard_scaler.fit_transform(test_pre)

# data split
X_train, X_valid, Y_train, Y_valid = train_test_split(train_pre, train_raw_labels, test_size=0.3, random_state=1)


# hyperparameters
input_layer = train_pre.shape[1]
output_layer = 2
num_epochs = 5001
learning_rate = 0.01
train_size = 0.8
layers_dims = [input_layer, 500, 500, output_layer]


parameters, submission_name = model(X_train, Y_train, X_valid, Y_valid, layers_dims, num_epochs=num_epochs,
                                    learning_rate=learning_rate, print_cost=False, plot_cost=True, l2_beta=0.1,
                                    keep_prob=0.5, minibatch_size=0, return_best=True, print_accuracy=False,
                                    plot_accuracy=True)

print(submission_name)
final_prediction = predict(test_pre, parameters)
output_submission(test.index.values, final_prediction, 'PassengerId', 'Survived', submission_name)
