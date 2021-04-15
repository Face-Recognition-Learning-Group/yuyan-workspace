class SingleTon(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class DataLoader(metaclass=SingleTon):
    def __init__(self): 
        pass

    def load_mnist_data(self, preprocessing):
        if not self.exists():
            import Data.mnist_data as data_mnist
            self.x_train, self.y_train, self.x_test, self.y_test = data_mnist.load_data()
            self.x_train, self.x_test = preprocessing(self.x_train), preprocessing(self.x_test)
        return self.x_train, self.y_train, self.x_test, self.y_test


    def load_cifar10_data(self, data_dir, preprocessing):
        if not self.exists():
            import Data.cifar_10_data as data_cifar10
            self.x_train, self.y_train, self.x_test, self.y_test = data_cifar10.load_data(data_dir)
            self.x_train, self.x_test = preprocessing(self.x_train), preprocessing(self.x_test)
        return self.x_train, self.y_train, self.x_test, self.y_test
            


    def exists(self):
        if hasattr(self, 'x_train'):
            return True
        else:
            return False