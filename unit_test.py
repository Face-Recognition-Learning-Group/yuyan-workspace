if __name__ == '__main__':
    import Data
    x_train, y_train, x_test, y_test = Data.load_data("mnist")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train, y_train, x_test, y_test = Data.load_data("cifar10", data_dir='./tmp/cifar-10-batches-py')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train, y_train, x_test, y_test = Data.load_data("cifar100", data_dir='./tmp/cifar-100-python')
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    