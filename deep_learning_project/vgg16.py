import keras
from keras.models import *
from keras.layers import *


pretrained_url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def get_vgg_encoder(input_height=224, input_width=224, pretrained='imagenet', channels=3):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, channels))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_last')(x)
    f1 = x

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_last')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_last')(x)
    f2 = x

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_last')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_last')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_last')(x)
    f3 = x

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_last')(x)
    f4 = x

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_last')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_last')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_last')(x)
    f5 = x

    if pretrained == 'imagenet':
        VGG_Weights_path = keras.utils.get_file(pretrained_url.split('/')[-1], pretrained_url)
        Model(img_input, x).load_weights(VGG_Weights_path, by_name=True, skip_mismatch=True)

    return img_input, [f1, f2, f3, f4, f5]
