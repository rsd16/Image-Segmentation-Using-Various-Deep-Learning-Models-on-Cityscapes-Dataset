import numpy as np
import keras
from keras.models import *
from keras.layers import *
import keras.backend as K

from .model_utils import get_segmentation_model, resize_image
from .vgg16 import get_vgg_encoder
from .basic_model import vanilla_encoder
from .resnet50 import get_resnet50_encoder


def pool_block(feats, pool_factor):
    h = K.int_shape(feats)[1]
    w = K.int_shape(feats)[2]

    pool_size = strides = [int(np.round(float(h) / pool_factor)), int(np.round(float(w) / pool_factor))]

    x = AveragePooling2D(pool_size, data_format='channels_last', strides=strides, padding='same')(feats)
    x = Conv2D(512, (1, 1), data_format='channels_last', padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = resize_image(x, strides, data_format='channels_last')

    return x

def base_pspnet(n_classes, encoder,  input_height=384, input_width=576, channels=3):
    assert input_height % 192 == 0
    assert input_width % 192 == 0

    img_input, levels = encoder(input_height=input_height,  input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]

    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)

    o = Concatenate(axis=-1)(pool_outs)

    o = Conv2D(512, (1, 1), data_format='channels_last', use_bias=False, name='seg_feats')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(n_classes, (3, 3), data_format='channels_last', padding='same')(o)
    o = resize_image(o, (8, 8), data_format='channels_last')

    model = get_segmentation_model(img_input, o)

    return model

def pspnet(n_classes, input_height=384, input_width=576, channels=3):
    model = base_pspnet(n_classes, vanilla_encoder, input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = 'pspnet'
    return model

def vgg_pspnet(n_classes, input_height=384, input_width=576, channels=3):
    model = base_pspnet(n_classes, get_vgg_encoder, input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = 'vgg_pspnet'
    return model

def resnet50_pspnet(n_classes, input_height=384, input_width=576, channels=3):
    model = base_pspnet(n_classes, get_resnet50_encoder, input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = 'resnet50_pspnet'
    return model

def mobilenet_pspnet(n_classes, input_height=224, input_width=224):
    model = base_pspnet(n_classes, get_mobilenet_encoder, input_height=input_height, input_width=input_width)
    model.model_name = 'mobilenet_pspnet'
    return model

if __name__ == '__main__':
    m = base_pspnet(101, vanilla_encoder)
    m = base_pspnet(101, get_mobilenet_encoder, True, 224, 224)
    m = base_pspnet(101, get_vgg_encoder)
    m = base_pspnet(101, get_resnet50_encoder)
