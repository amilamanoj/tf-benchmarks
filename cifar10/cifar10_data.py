from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils import to_categorical

num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


def generator_train():
    i = 0
    while (True):

        x = x_train[i]
        y = y_train[i]

        i += 1
        if i == x_train.shape[0]:
            i = 0
        yield x, y


def generator_test():
    i = 0
    while (True):

        x = x_test[i]
        y = y_test[i]
        i += 1
        if i == x_test.shape[0]:
            i = 0
        yield x, y


def train_len():
    return len(x_train)


def val_len():
    return len(x_test)
