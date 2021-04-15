import tensorflow as tf


def multi_layer_perception(dataset_name, input_shape=(28,28), activation='relu'):
    from Data import get_class_num
    class_number = get_class_num(dataset_name)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(256, activation=activation),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(class_number, activation='softmax')
    ])
    return model

def resNet50(dataset_name, input_shape=(32,32,3)):
    from Data import get_class_num
    class_number = get_class_num(dataset_name)
    model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_tensor=None,
            input_shape=input_shape,
            pooling="avg",
            classes=class_number,
        )
    return model

def MobileNetV2(dataset_name, input_shape=(32,32,3)):
    from Data import get_class_num
    class_number = get_class_num(dataset_name)
    model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=False,
            weights=None,
            input_tensor=None,
            pooling=None,
            classes=class_number,
            classifier_activation="softmax",
        )
    return model

