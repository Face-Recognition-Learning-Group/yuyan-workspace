import time
import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.keras.callbacks import LearningRateScheduler

def mnist_classfication_train_procedure(save_path, epoch=5):
    import Data
    import Model

    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name="mnist")
    end_time = time.time()
    print("[Train] Load data time: ", end_time - start_time)
    
    model = Model.multi_layer_perception(dataset_name="mnist", activation="relu")
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epoch)
    model.save(save_path)

def cifar10_classfication_train_procedure(save_path, epoch=5):
    import Data
    import Model

    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name="cifar10",data_dir="./tmp/cifar-10-batches-py/")
    end_time = time.time()
    print("[Train] Load data time: ", end_time - start_time)
    
    model = Model.multi_layer_perception(dataset_name="cifar10", input_shape=(32, 32, 3), activation="relu")
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epoch)
    model.save(save_path)
    
def cifar10_classfication_train_procedure_with_resNet50(save_path, epoch=5):
    import Data
    import Model

    def lr_schedule(epoch_index, cur_lr):
        if epoch_index < 1:
            return 1e-4
        elif epoch_index < 60:
            return 1e-3
        elif epoch_index < 100: 
            return 0.02
        elif epoch_index < 120: 
            return 0.02
        else:
            return 0.02
    
    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name="cifar10",data_dir="./tmp/cifar-10-batches-py/")
    end_time = time.time()
    print("[Train] Load data time: ", end_time - start_time)
    
    model = Model.resNet50(dataset_name="cifar10", input_shape=(32, 32, 3))
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epoch, batch_size=64, callbacks=[lr_scheduler])
    model.save(save_path)

def cifar10_classfication_train_procedure_with_resNet50V2(save_path, epoch=5):
    import Data
    import Model

    def lr_schedule(epoch_index, cur_lr):
        if epoch_index < 1:
            return 1e-4
        elif epoch_index < 60:
            return 1e-3
        elif epoch_index < 100: 
            return 0.02
        elif epoch_index < 120: 
            return 0.02
        else:
            return 0.02
    
    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name="cifar10",data_dir="./tmp/cifar-10-batches-py/")
    end_time = time.time()
    print("[Train] Load data time: ", end_time - start_time)
    
    model = Model.resNet50V2(dataset_name="cifar10", input_shape=(32, 32, 3))
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epoch, batch_size=64, callbacks=[lr_scheduler])
    model.save(save_path)
   
def cifar10_classfication_train_procedure_with_MobileNet(save_path, epoch=5):
    import Data
    import Model
    
    def lr_schedule(epoch_index, cur_lr):
        if epoch_index < 1:
            return 1e-4
        elif epoch_index < 60:
            return 1e-3
        elif epoch_index < 100: 
            return 0.02
        elif epoch_index < 120: 
            return 0.02
        else:
            return 0.02
    
    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name="cifar10",data_dir="./tmp/cifar-10-batches-py/")
    end_time = time.time()
    print("[Train] Load data time: ", end_time - start_time)
    
    model = Model.MobileNet(dataset_name="cifar10", input_shape=(32, 32, 3))
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epoch, batch_size=64, callbacks=[lr_scheduler])
    model.save(save_path)

def cifar10_classfication_train_procedure_with_MobileNetV2(save_path, epoch=5):
    import Data
    import Model

    # def lr_schedule(epoch_index, cur_lr):
    #     if epoch_index < 1:
    #         return 1e-4
    #     elif epoch_index < 60:
    #         return 1e-3
    #     elif epoch_index < 100: 
    #         return 1e-4
    #     elif epoch_index < 120: 
    #         return 1e-5
    #     else:
    #         return 1e-6
    
    def lr_schedule(epoch_index, cur_lr):
        if epoch_index < 1:
            return 1e-4
        elif epoch_index < 60:
            return 1e-3
        elif epoch_index < 100: 
            return 0.02
        elif epoch_index < 120: 
            return 0.02
        else:
            return 0.02
    
    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name="cifar10",data_dir="./tmp/cifar-10-batches-py/")
    end_time = time.time()
    print("[Train] Load data time: ", end_time - start_time)
    
    model = Model.MobileNetV2(dataset_name="cifar10", input_shape=(32, 32, 3))
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epoch, batch_size=64, callbacks=[lr_scheduler])
    model.save(save_path)
    
def cifar10_classfication_train_procedure_with_DenseNet121(save_path, epoch=5):
    import Data
    import Model
  
    
    def lr_schedule(epoch_index, cur_lr):
        if epoch_index < 1:
            return 1e-4
        elif epoch_index < 60:
            return 1e-3
        elif epoch_index < 100: 
            return 0.02
        elif epoch_index < 120: 
            return 0.02
        else:
            return 0.02
    
    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name="cifar10",data_dir="./tmp/cifar-10-batches-py/")
    end_time = time.time()
    print("[Train] Load data time: ", end_time - start_time)
    
    model = Model.DenseNet121(dataset_name="cifar10", input_shape=(32, 32, 3))
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epoch, batch_size=64, callbacks=[lr_scheduler])
    model.save(save_path)
    
    
def cifar100_classfication_train_procedure_with_resNet50(save_path, epoch=5):
    import Data
    import Model

    def lr_schedule(epoch_index, cur_lr):
        if epoch_index < 1:
            return 1e-4
        elif epoch_index < 60:
            return 1e-3
        elif epoch_index < 100: 
            return 0.02
        elif epoch_index < 120: 
            return 0.02
        else:
            return 0.02
    
    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name="cifar100",data_dir="./tmp/cifar-100-python/")
    end_time = time.time()
    print("[Train] Load data time: ", end_time - start_time)
    
    model = Model.resNet50(dataset_name="cifar100", input_shape=(32, 32, 3))
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epoch, batch_size=64, callbacks=[lr_scheduler])
    model.save(save_path)
    

def cifar100_classfication_train_procedure_with_resNet50V2(save_path, epoch=5):
    import Data
    import Model

    def lr_schedule(epoch_index, cur_lr):
        if epoch_index < 1:
            return 1e-4
        elif epoch_index < 60:
            return 1e-3
        elif epoch_index < 100: 
            return 0.02
        elif epoch_index < 120: 
            return 0.02
        else:
            return 0.02
    
    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name="cifar100",data_dir="./tmp/cifar-100-python/")
    end_time = time.time()
    print("[Train] Load data time: ", end_time - start_time)
    
    model = Model.resNet50V2(dataset_name="cifar100", input_shape=(32, 32, 3))
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epoch, batch_size=64, callbacks=[lr_scheduler])
    model.save(save_path)


def cifar100_classfication_train_procedure_with_MobileNet(save_path, epoch=5):
    import Data
    import Model

    def lr_schedule(epoch_index, cur_lr):
        if epoch_index < 1:
            return 1e-4
        elif epoch_index < 60:
            return 1e-3
        elif epoch_index < 100: 
            return 0.02
        elif epoch_index < 120: 
            return 0.02
        else:
            return 0.02
    
    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name="cifar100",data_dir="./tmp/cifar-100-python/")
    end_time = time.time()
    print("[Train] Load data time: ", end_time - start_time)
    
    model = Model.MobileNet(dataset_name="cifar100", input_shape=(32, 32, 3))
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epoch, batch_size=64, callbacks=[lr_scheduler])
    model.save(save_path)
    

def cifar100_classfication_train_procedure_with_MobileNetV2(save_path, epoch=5):
    import Data
    import Model

    def lr_schedule(epoch_index, cur_lr):
        if epoch_index < 1:
            return 1e-4
        elif epoch_index < 60:
            return 1e-3
        elif epoch_index < 100: 
            return 0.02
        elif epoch_index < 120: 
            return 0.02
        else:
            return 0.02
    
    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name="cifar100",data_dir="./tmp/cifar-100-python/")
    end_time = time.time()
    print("[Train] Load data time: ", end_time - start_time)
    
    model = Model.MobileNetV2(dataset_name="cifar100", input_shape=(32, 32, 3))
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epoch, batch_size=64, callbacks=[lr_scheduler])
    model.save(save_path)


def cifar100_classfication_train_procedure_with_DenseNet121(save_path, epoch=5):
    import Data
    import Model

    def lr_schedule(epoch_index, cur_lr):
        if epoch_index < 1:
            return 1e-4
        elif epoch_index < 60:
            return 1e-3
        elif epoch_index < 100: 
            return 0.02
        elif epoch_index < 120: 
            return 0.02
        else:
            return 0.02
    
    start_time = time.time()
    x_train, y_train, x_test, y_test = Data.load_data(dataset_name="cifar100",data_dir="./tmp/cifar-100-python/")
    end_time = time.time()
    print("[Train] Load data time: ", end_time - start_time)
    
    model = Model.DenseNet121(dataset_name="cifar100", input_shape=(32, 32, 3))
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=epoch, batch_size=64, callbacks=[lr_scheduler])
    model.save(save_path)