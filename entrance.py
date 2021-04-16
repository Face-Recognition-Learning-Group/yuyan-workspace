import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import Tasks.train as train_porcedure
import Tasks.test as test_porcedure

if __name__ == '__main__':
    
    #*************************************************************************************
    #                  MNIST Train & Test
    #
    #*************************************************************************************
    
    # save_path = "./tmp/model/mnist/"
    # print("Train MNIST with mnist_classfication_train_procedure")
    # train_porcedure.mnist_classfication_train_procedure(save_path=save_path, epoch=5)
    # test_porcedure.mnist_classification_test_procedure(save_path=save_path)
    
    
    #*************************************************************************************
    #                  CIFAR10 Train & Test
    #
    #*************************************************************************************
    
    # save_path = "./tmp/model/cifar10_model/"
    # print("*"*80)
    # print("Train CIFAR10 with cifar10_classfication_train_procedure")
    # train_porcedure.cifar10_classfication_train_procedure(save_path=save_path, epoch=5)
    # test_porcedure.cifar10_classification_test_procedure(save_path=save_path)

    # save_path = "./tmp/model/cifar10_alexnet50_model/"
    # print("*"*80)
    # print("Train CIFAR10 with cifar10_classfication_train_procedure_with_resNet50")
    # train_porcedure.cifar10_classfication_train_procedure_with_alexnet(save_path=save_path, epoch=30)
    # test_porcedure.cifar10_classification_test_procedure(save_path=save_path)

    save_path = "./tmp/model/cifar10_resnet34_model/"
    print("*"*80)
    print("Train CIFAR10 with cifar10_classfication_train_procedure_with_resNet34")
    train_porcedure.cifar10_classfication_train_procedure_with_resnet34(save_path=save_path, epoch=30)
    test_porcedure.cifar10_classification_test_procedure(save_path=save_path)
    
    # save_path = "./tmp/model/cifar10_restNet50V2_model/"
    # print("*"*80)
    # print("Train CIFAR10 with cifar10_classfication_train_procedure_with_resNet50V2")
    # train_porcedure.cifar10_classfication_train_procedure_with_resNet50V2(save_path=save_path, epoch=130)
    # test_porcedure.cifar10_classification_test_procedure(save_path=save_path)
     
    # save_path = "./tmp/model/cifar10__MobileNet_model/"
    # print("*"*80)
    # print("Train CIFAR10 with cifar10_classfication_train_procedure_with_MobileNet")
    # train_porcedure.cifar10_classfication_train_procedure_with_MobileNet(save_path=save_path, epoch=130)
    # test_porcedure.cifar10_classification_test_procedure(save_path=save_path)
    
    # save_path = "./tmp/model/cifar10__MobileNetV2_model/"
    # print("*"*80)
    # print("Train CIFAR10 with cifar10_classfication_train_procedure_with_MobileNetV2")
    # train_porcedure.cifar10_classfication_train_procedure_with_MobileNetV2(save_path=save_path, epoch=130)
    # test_porcedure.cifar10_classification_test_procedure(save_path=save_path)
    
    # save_path = "./tmp/model/cifar10__DenseNet121_model/"
    # print("*"*80)
    # print("Train CIFAR10 with cifar10_classfication_train_procedure_with_DenseNet121")
    # train_porcedure.cifar10_classfication_train_procedure_with_DenseNet121(save_path=save_path, epoch=130)
    # test_porcedure.cifar10_classification_test_procedure(save_path=save_path)
    
    #*************************************************************************************
    #                  CIFAR100 Train & Test
    #
    #*************************************************************************************
    
    # save_path = "./tmp/model/cifar100_restNet50_model/"
    # print("*"*80)
    # print("Train CIFAR100 with cifar100_classfication_train_procedure_with_resNet50")
    # train_porcedure.cifar100_classfication_train_procedure_with_resNet50(save_path=save_path, epoch=130)
    # test_porcedure.cifar100_classification_test_procedure(save_path=save_path)
    
    # save_path = "./tmp/model/cifar100_restNet50V2_model/"
    # print("*"*80)
    # print("Train CIFAR100 with cifar100_classfication_train_procedure_with_resNet50V2")
    # train_porcedure.cifar100_classfication_train_procedure_with_resNet50V2(save_path=save_path, epoch=130)
    # test_porcedure.cifar100_classification_test_procedure(save_path=save_path)
     
    # save_path = "./tmp/model/cifar100__MobileNet_model/"
    # print("*"*80)
    # print("Train CIFAR100 with cifar100_classfication_train_procedure_with_MobileNet")
    # train_porcedure.cifar100_classfication_train_procedure_with_MobileNet(save_path=save_path, epoch=130)
    # test_porcedure.cifar100_classification_test_procedure(save_path=save_path)
    
    # save_path = "./tmp/model/cifar100__MobileNetV2_model/"
    # print("*"*80)
    # print("Train CIFAR100 with cifar100_classfication_train_procedure_with_MobileNetV2")
    # train_porcedure.cifar100_classfication_train_procedure_with_MobileNetV2(save_path=save_path, epoch=130)
    # test_porcedure.cifar100_classification_test_procedure(save_path=save_path)
    
    # save_path = "./tmp/model/cifar100__DenseNet121_model/"
    # print("*"*80)
    # print("Train CIFAR100 with cifar100_classfication_train_procedure_with_DenseNet121")
    # train_porcedure.cifar100_classfication_train_procedure_with_DenseNet121(save_path=save_path, epoch=130)
    # test_porcedure.cifar100_classification_test_procedure(save_path=save_path)
