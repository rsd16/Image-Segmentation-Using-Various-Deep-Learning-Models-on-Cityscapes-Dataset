from keras.models import *
from keras.layers import *
import keras.backend as K


def vanilla_encoder(input_height=224, input_width=224, channels=3):
    img_input = Input(shape=(input_height, input_width, channels))

    x = img_input
    levels = []

    x = (ZeroPadding2D((1, 1), data_format='channels_last'))(x)
    x = (Conv2D(64, (3, 3), data_format='channels_last', padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2), data_format='channels_last'))(x)
    levels.append(x)

    x = (ZeroPadding2D((1, 1), data_format='channels_last'))(x)
    x = (Conv2D(128, (3, 3), data_format='channels_last', padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((2, 2), data_format='channels_last'))(x)
    levels.append(x)

    for _ in range(3):
        x = (ZeroPadding2D((1, 1), data_format='channels_last'))(x)
        x = (Conv2D(256, (3, 3), data_format='channels_last', padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((2, 2), data_format='channels_last'))(x)
        levels.append(x)

    return img_input, levels
