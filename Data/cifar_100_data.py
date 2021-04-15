import os
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(dir_path):
    train_path = os.path.join(dir_path, "train")
    test_path = os.path.join(dir_path, "test")

    train_d = unpickle(train_path)
    test_d = unpickle(test_path)
    
    x_train, y_train , x_test, y_test = [], [], [], []

    for each_label, each_data in zip(train_d[b'fine_labels'], train_d[b'data']):
        y_train.append(each_label)
        x_train.append(np.transpose(each_data.reshape((3, 32, 32)), (1, 2, 0)))
    for each_label, each_data in zip(test_d[b'fine_labels'], test_d[b'data']):
        y_test.append(each_label)
        x_test.append(np.transpose(each_data.reshape((3, 32, 32)), (1, 2, 0)))
    
    x_train, x_test = np.array(x_train), np.array(x_test)
    y_train, y_test = np.array(y_train), np.array(y_test)
    return x_train, y_train, x_test, y_test


def get_class_num():
    return 100

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data("../tmp/cifar-100-py/")
    print(x_train.shape, len(y_train), x_test.shape, len(y_test))