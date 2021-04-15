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
        if not self.exists(has_key="mnist_x_train"):
            import Data.mnist_data as data_mnist
            self.mnist_x_train, self.y_train, self.x_test, self.y_test = data_mnist.load_data()
            self.mnist_x_train, self.x_test = preprocessing(self.mnist_x_train), preprocessing(self.x_test)
            self.delete_other_keys(has_key="mnist_x_train")
        return self.mnist_x_train, self.y_train, self.x_test, self.y_test


    def load_cifar10_data(self, data_dir, preprocessing):
        if not self.exists(has_key="cifar10_x_train"):
            import Data.cifar_10_data as data_cifar10
            self.cifar10_x_train, self.y_train, self.x_test, self.y_test = data_cifar10.load_data(data_dir)
            self.cifar10_x_train, self.x_test = preprocessing(self.cifar10_x_train), preprocessing(self.x_test)
            self.delete_other_keys(has_key="cifar10_x_train")
        return self.cifar10_x_train, self.y_train, self.x_test, self.y_test
    
    
    def load_cifar100_data(self, data_dir, preprocessing):
        if not self.exists(has_key="cifar100_x_train"):
            import Data.cifar_100_data as data_cifar100
            self.cifar100_x_train, self.y_train, self.x_test, self.y_test = data_cifar100.load_data(data_dir)
            self.cifar100_x_train, self.x_test = preprocessing(self.cifar100_x_train), preprocessing(self.x_test)
            self.delete_other_keys(has_key="cifar100_x_train")
        return self.cifar100_x_train, self.y_train, self.x_test, self.y_test


    def exists(self, has_key):
        if hasattr(self, has_key):
            return True
        else:
            return False
    
    
    def delete_other_keys(self, has_key):
        all_possible_keys = ["mnist_x_train", "cifar10_x_train", "cifar100_x_train"]
        for key in all_possible_keys:
            if key == has_key:
                continue
            elif hasattr(self, key):
                # print("delete data of ", key)
                delattr(self, key)
        
        