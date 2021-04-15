import time
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
    print("[Test] Load data time: ", time.time() - start_time)
    model = tf.keras.models.load_model(save_path)
    model.evaluate(x_test, y_test)

def cifar10_classification_test_procedure_with_restNet50(save_path):
    import Data

    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name='cifar10', data_dir="./tmp/cifar-10-batches-py/")
    print("[Test] Load data time: ", time.time() - start_time)
    model = tf.keras.models.load_model(save_path)
    model.evaluate(x_test, y_test)
    
def cifar10_classification_test_procedure_with_MobileNetV2(save_path):
    import Data

    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name='cifar10', data_dir="./tmp/cifar-10-batches-py/")
    print("[Test] Load data time: ", time.time() - start_time)
    model = tf.keras.models.load_model(save_path)
    model.evaluate(x_test, y_test)