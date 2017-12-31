from dataset import load_data, convert_to_one_hot, output_submission
from model import model
from methods import predict


TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'

train, test = load_data(TRAIN_PATH, TEST_PATH)

CLASSES = 2
train_dataset_size = train.shape[0]
train_raw_labels = train.Survived.values

# pre-process
train_pre = train.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'Age', 'Sex'], axis=1).values
test_pre = test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked', 'Age', 'Sex'], axis=1).values

# The labels need to be one-hot encoded
train_labels = convert_to_one_hot(train_dataset_size, train_raw_labels, CLASSES)

trained_parameters, submission_name = model(train_pre, train_labels, train_size=0.7,
                                            num_epochs=5001, learning_rate=0.0001)

final_prediction = predict(test_pre, trained_parameters)

output_submission(test.PassengerId.values, final_prediction, 'PassengerId', 'Survived', submission_name)