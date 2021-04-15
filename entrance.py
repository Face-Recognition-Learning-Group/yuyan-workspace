import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import Tasks.train as train_porcedure
import Tasks.test as test_porcedure

if __name__ == '__main__':
    # save_path = "./tmp/mnist/"
    # train_porcedure.mnist_classfication_train_procedure(save_path=save_path, epoch=5)
    # test_porcedure.mnist_classification_test_procedure(save_path=save_path)
    # save_path = "./tmp/cifar_model/"
    # train_porcedure.cifar10_classfication_train_procedure(save_path=save_path, epoch=5)
    # test_porcedure.cifar10_classification_test_procedure(save_path=save_path)
    # save_path = "./tmp/cifar_restNet50_model/"
    # train_porcedure.cifar10_classfication_train_procedure_with_resNet50(save_path=save_path, epoch=130)
    # test_porcedure.cifar10_classification_test_procedure_with_restNet50(save_path=save_path)
    save_path = "./tmp/cifar__MobileNetV2_model/"
    train_porcedure.cifar10_classfication_train_procedure_with_MobileNetV2(save_path=save_path, epoch=130)
    test_porcedure.cifar10_classification_test_procedure_with_MobileNetV2(save_path=save_path)
