import tensorflow as tf

import Data
import cv2

def image_preprocessing(images):
    return images

if __name__ == '__main__':
	x_train, y_train, x_test, y_test = Data.load_data('cifar10', data_dir='./tmp/cifar-10-batches-py', preprocessing=image_preprocessing)

	test_image = cv2.cvtColor(x_test[2], cv2.COLOR_RGB2BGR)
	print(y_test[2])
	cv2.imwrite('./dbg.jpg', test_image)