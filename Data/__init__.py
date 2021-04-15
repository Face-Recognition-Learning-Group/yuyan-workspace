import Data.exceptions as data_exceptions
import Data.preprocessing as data_preprocessing


def load_data(dataset_name, preprocessing=data_preprocessing.default_uint8_image_preprocessing, data_dir=None):
    assert dataset_name in ['mnist', 'MNIST', "cifar10", "CIFAR10", "cifar-10", "CIFAR-10"]
    if dataset_name == "mnist" or dataset_name == "MNIST":
        try:
            import Data.Instances as data_instances
            x_train, y_train, x_test, y_test = data_instances.DataLoader().load_mnist_data(preprocessing)
            return x_train, y_train, x_test, y_test
          
        except Exception as e:
                raise data_exceptions.DataLoadError("*** MNIST Data loaded/preporcessing derror. Infomation: {}".format(str(e)))

    # expanding
    if dataset_name == "cifar10" or dataset_name == "CIFAR10" or \
        dataset_name == "cifar-10" or dataset_name == "CIFAR-10":
        assert data_dir is not None
        try:
            import Data.Instances as data_instances
            x_train, y_train, x_test, y_test = data_instances.DataLoader().load_cifar10_data(data_dir, preprocessing)
            return x_train, y_train, x_test, y_test
    
        except Exception as e:
                raise data_exceptions.DataLoadError("*** CIFAR10 Data loaded/preporcessing derror. Infomation: {}".format(str(e)))

    raise data_exceptions.DataLoadError("*** Unexpected Error. dataset_name: {}".format(dataset_name))


def get_class_num(dataset_name):
    assert dataset_name in ['mnist', 'MNIST', "cifar10", "CIFAR10", "cifar-10", "CIFAR-10"]
    
    if dataset_name == 'mnist' or dataset_name == 'MNIST':
        try:
            import Data.mnist_data as data_mnist
            return data_mnist.get_class_num()
        except Exception as e:
            raise data_exceptions.DataLoadError("*** MNIST class number loaded error. Infomation: ", str(e))


    if dataset_name == "cifar10" or dataset_name == "CIFAR10" or \
        dataset_name == "cifar-10" or dataset_name == "CIFAR-10":
        try:
            import Data.cifar_10_data as data_cifar_10
            return data_cifar_10.get_class_num()
        except Exception as e:
            raise data_exceptions.DataLoadError("*** CIFAR class number loaded error. Infomation: ", str(e))
    raise data_exceptions.DataLoadError("*** Unexpected dataset_name: {}".format(dataset_name))