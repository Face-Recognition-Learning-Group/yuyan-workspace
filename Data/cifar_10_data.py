import os
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(dir_path):
    train_1 = os.path.join(dir_path, "data_batch_1")
    train_2 = os.path.join(dir_path, "data_batch_2")
    train_3 = os.path.join(dir_path, "data_batch_3")
    train_4 = os.path.join(dir_path, "data_batch_4")
    train_5 = os.path.join(dir_path, "data_batch_5")
    test = os.path.join(dir_path, "test_batch")

    train_d_1 = unpickle(train_1)
    train_d_2 = unpickle(train_2)
    train_d_3 = unpickle(train_3)
    train_d_4 = unpickle(train_4)
    train_d_5 = unpickle(train_5)
    test_d = unpickle(test)
    
    x_train, y_train , x_test, y_test = [], [], [], []

    for each_label, each_data in zip(train_d_1[b'labels'], train_d_1[b'data']):
        y_train.append(each_label)
        x_train.append(np.transpose(each_data.reshape((3, 32, 32)), (2, 1, 0)))
    for each_label, each_data in zip(train_d_2[b'labels'], train_d_2[b'data']):
        y_train.append(each_label)
        x_train.append(np.transpose(each_data.reshape((3, 32, 32)), (2, 1, 0)))
    for each_label, each_data in zip(train_d_3[b'labels'], train_d_3[b'data']):
        y_train.append(each_label)
        x_train.append(np.transpose(each_data.reshape((3, 32, 32)), (2, 1, 0)))
    for each_label, each_data in zip(train_d_4[b'labels'], train_d_4[b'data']):
        y_train.append(each_label)
        x_train.append(np.transpose(each_data.reshape((3, 32, 32)), (2, 1, 0)))
    for each_label, each_data in zip(train_d_5[b'labels'], train_d_5[b'data']):
        y_train.append(each_label)
        x_train.append(np.transpose(each_data.reshape((3, 32, 32)), (2, 1, 0)))
    for each_label, each_data in zip(test_d[b'labels'], test_d[b'data']):
        y_test.append(each_label)
        x_test.append(np.transpose(each_data.reshape((3, 32, 32)), (2, 1, 0)))
    
    x_train, x_test = np.array(x_train), np.array(x_test)
    y_train, y_test = np.array(y_train), np.array(y_test)
    return x_train, y_train, x_test, y_test


def get_class_num():
    return 10

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data("../tmp/cifar-10-batches-py/")
    print(x_train.shape, len(y_train), x_test.shape, len(y_test))