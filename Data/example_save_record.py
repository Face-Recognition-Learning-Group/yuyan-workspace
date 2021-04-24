import os
import traceback
import numpy as np
import tensorflow as tf

from Data.preprocessing import default_uint8_image_preprocessing

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_train_data(dir_path, save_filename):
    assert os.path.exists(os.path.dirname(save_filename))

    try:
        train_path = os.path.join(dir_path, "train")
        train_d = unpickle(train_path)
        writer = tf.io.TFRecordWriter(save_filename)
        for each_label, each_data in zip(train_d[b'fine_labels'], train_d[b'data']):
            each_image = np.transpose(each_data.reshape((3, 32, 32)), (2, 1, 0))
            each_image = default_uint8_image_preprocessing(each_image)
            each_image_bytes = each_image.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[each_image_bytes])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[each_label]))
            }))
            writer.write(example.SerializeToString())
        writer.close()
        return True
    except Exception as e:
        traceback.print_exc()
        return False

def save_test_data(dir_path, save_filename):
    assert os.path.exists(os.path.dirname(save_filename))

    try:
        test_path = os.path.join(dir_path, "test")
        test_d = unpickle(test_path)
        writer = tf.io.TFRecordWriter(save_filename)
        for each_label, each_data in zip(test_d[b'fine_labels'], test_d[b'data']):
            each_image = np.transpose(each_data.reshape((3, 32, 32)), (2, 1, 0))
            each_image = default_uint8_image_preprocessing(each_image)
            print(each_image.dtype)
            each_image_bytes =each_image.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[each_image_bytes])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[each_label]))
            }))
            writer.write(example.SerializeToString())
        writer.close()
        return True
    except Exception as e:
        traceback.print_exc()
        return False



if __name__ == '__main__':
    # print(__file__)
    # assert save_train_data(dir_path=r'D:\yuyan-workspace\tmp\cifar-100-python',
    #                 save_filename=r'D:\yuyan-workspace\tmp\cifar_100_train_records')

    assert save_test_data(dir_path=r'D:\yuyan-workspace\tmp\cifar-100-python',
                    save_filename=r'D:\yuyan-workspace\tmp\cifar_100_test_records')

