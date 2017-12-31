from dataset import load_data, convert_to_one_hot, generate_train_subsets
from model import model


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

train_set, test_set = generate_train_subsets(train_pre)
train_labels, test_labels = generate_train_subsets(train_labels)

trained_parameters = model(train_set, train_labels, test_set, test_labels, test, test_pre,
                           num_epochs=1001, learning_rate=0.0001)
