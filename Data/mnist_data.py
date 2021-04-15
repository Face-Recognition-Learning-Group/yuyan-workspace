import tensorflow as tf

def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    

    return x_train, y_train, x_test, y_test

def get_class_num():
    return 10