import Data.exceptions as data_exceptions
import Data.preprocessing as data_preprocessing


def load_data(dataset_name, preprocessing=data_preprocessing.default_uint8_image_preprocessing, data_dir=None, record_file=None,
              batch_size=None, parallel_num=None):
    assert dataset_name in ['mnist', 'MNIST', "cifar10", "CIFAR10", "cifar-10", "CIFAR-10", "cifar100", "CIFAR100", "cifar-100", "CIFAR-100",
                            'faces_vgg', 'faces_vgg']
    # mnist
    if dataset_name == "mnist" or dataset_name == "MNIST":
        try:
            import Data.Instances as data_instances
            x_train, y_train, x_test, y_test = data_instances.DataLoader().load_mnist_data(preprocessing)
            return x_train, y_train, x_test, y_test
          
        except Exception as e:
                raise data_exceptions.DataLoadError("*** MNIST Data loaded/preporcessing derror. Infomation: {}".format(str(e)))

    # cifar10
    if dataset_name == "cifar10" or dataset_name == "CIFAR10" or \
        dataset_name == "cifar-10" or dataset_name == "CIFAR-10":
        assert data_dir is not None
        try:
            import Data.Instances as data_instances
            x_train, y_train, x_test, y_test = data_instances.DataLoader().load_cifar10_data(data_dir, preprocessing)
            return x_train, y_train, x_test, y_test
    
        except Exception as e:
                raise data_exceptions.DataLoadError("*** CIFAR-10 Data loaded/preporcessing derror. Infomation: {}".format(str(e)))
    
    # cifar100
    if dataset_name == "cifar100" or dataset_name == "CIFAR100" or \
        dataset_name == "cifar-100" or dataset_name == "CIFAR-100":
        assert data_dir is not None
        try:
            import Data.Instances as data_instances
            x_train, y_train, x_test, y_test = data_instances.DataLoader().load_cifar100_data(data_dir, preprocessing)
            return x_train, y_train, x_test, y_test
    
        except Exception as e:
            raise data_exceptions.DataLoadError("*** CIFAR-100 Data loaded/preporcessing derror. Infomation: {}".format(str(e)))

    # faces_vgg
    if dataset_name == 'faces_vgg' or dataset_name == 'face_vgg':
        assert record_file is not None
        try:
            from Data.load_tfrecords import _load_tfrecord
            train_record_tf_object = _load_tfrecord(record_file, batch_size=batch_size, parallel_batches=parallel_num)
            return train_record_tf_object
        except Exception as e:
            raise data_exceptions.DataLoadError(
                "*** faces_vgg Data loaded/preporcessing derror. Infomation: {}".format(str(e)))
    raise data_exceptions.DataLoadError("*** Unexpected Error. dataset_name: {}".format(dataset_name))


def get_class_num(dataset_name):
    assert dataset_name in ['mnist', 'MNIST', "cifar10", "CIFAR10", "cifar-10", "CIFAR-10", "cifar100", "CIFAR100", "cifar-100", "CIFAR-100",
                            'faces_vgg', 'face_vgg']
    
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
            raise data_exceptions.DataLoadError("*** CIFAR-10 class number loaded error. Infomation: ", str(e))

    if dataset_name == "cifar100" or dataset_name == "CIFAR100" or \
        dataset_name == "cifar-100" or dataset_name == "CIFAR-100":
        try:
            import Data.cifar_100_data as data_cifar_100
            return data_cifar_100.get_class_num()
        except Exception as e:
            raise data_exceptions.DataLoadError("*** CIFAR-100 class number loaded error. Infomation: ", str(e))

    if dataset_name == "faces_vgg" or dataset_name == "face_vgg":
        try:
            return 85164
        except Exception as e:
            raise data_exceptions.DataLoadError("*** faces_vgg class number loaded error. Infomation: ", str(e))
        
    
    raise data_exceptions.DataLoadError("*** Unexpected dataset_name: {}".format(dataset_name))