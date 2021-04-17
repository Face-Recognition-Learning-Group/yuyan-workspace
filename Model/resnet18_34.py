
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2


def identity_block(input_tensor, kernel_size, filters, weight_decay=5e-4):
    filters1, filters2 = filters

    x = layers.Conv2D(filters1, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               strides=(2, 2),
               weight_decay=5e-4):
    filters1, filters2 = filters

    x = layers.Conv2D(filters1, kernel_size, strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    shortcut = layers.Conv2D(filters2, kernel_size, strides=strides,
                             padding='same',
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(weight_decay))(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet18(input_shape, class_number):
    weight_decay = 1e-4
    input_tensor = layers.Input(input_shape, name='resnet18_input')


    x = layers.Conv2D(64, (7, 7),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      strides=(2, 2))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = identity_block(x, (3,3), [48,64])
    x = identity_block(x, (3,3), [48,64])

    x = conv_block(x, (3,3), [96,128], strides=(2,2))
    x = identity_block(x, (3,3), [96,128])

    x = conv_block(x, (3,3), [128,256], strides=(2,2))
    x = identity_block(x, (3,3), [128,256])

    x = conv_block(x, (3,3), [256,512], strides=(2,2))
    x = identity_block(x, (3,3), [256,512])

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(class_number, activation='softmax', kernel_regularizer=l2(weight_decay))(x)

    model = Model(input_tensor, x, name='resnet18_model')
    return model

def ResNet34(input_shape, class_number):
    weight_decay = 1e-4
    input_tensor = layers.Input(input_shape, name='resnet34_input')


    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(weight_decay),
                      strides=(1, 1))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # x = layers.MaxPooling2D((6, 3), strides=(2, 2))(x)

    x = identity_block(x, (3,3), [48,64])
    x = identity_block(x, (3,3), [48,64])
    x = identity_block(x, (3,3), [48,64])

    x = conv_block(x, (3,3), [96,128], strides=(2,2))
    x = identity_block(x, (3,3), [96,128])
    x = identity_block(x, (3,3), [96,128])
    x = identity_block(x, (3,3), [96,128])

    x = conv_block(x, (3,3), [128,256], strides=(2,2))
    x = identity_block(x, (3,3), [128,256])
    x = identity_block(x, (3,3), [128,256])
    x = identity_block(x, (3,3), [128,256])
    x = identity_block(x, (3,3), [128,256])
    x = identity_block(x, (3,3), [128,256])

    x = conv_block(x, (3,3), [256,512], strides=(2,2))
    x = identity_block(x, (3,3), [256,512])
    x = identity_block(x, (3,3), [256,512])

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(class_number, activation='softmax', kernel_regularizer=l2(weight_decay))(x)

    model = Model(input_tensor, x, name='resnet34_model')
    return model
