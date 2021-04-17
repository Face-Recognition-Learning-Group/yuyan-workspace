import time
import numpy as np
import tensorflow as tf

def mnist_classification_test_procedure(save_path):
    import Data

    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name='mnist')
    print("[Test] Load data time: ", time.time() - start_time)

    model = tf.keras.models.load_model(save_path)
    model.evaluate(x_test, y_test)

def cifar10_classification_test_procedure(save_path):
    import Data

    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name='cifar10', data_dir="./tmp/cifar-10-batches-py/")
    # print("[Test] Load data time: ", time.time() - start_time)
    model = tf.keras.models.load_model(save_path)
    # model.evaluate(x_test, y_test)

    accuracy = []
    for each_test_matrix, real_label in zip(x_test, y_test):
        pred_onehot_label = model.predict(np.expand_dims(each_test_matrix, 0))[0]
        pred_label = np.argmax(pred_onehot_label, 0)
        print(pred_label, real_label)
        if int(pred_label) == int(real_label):
            accuracy.append(1)
        else:
            accuracy.append(0)
    print('Accuracy:', np.mean(accuracy))


def cifar100_classification_test_procedure(save_path):
    import Data
    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name='cifar100', data_dir="./tmp/cifar-100-python/")
    print("[Test] Load data time: ", time.time() - start_time)
    model = tf.keras.models.load_model(save_path)
    model.evaluate(x_test, y_test)