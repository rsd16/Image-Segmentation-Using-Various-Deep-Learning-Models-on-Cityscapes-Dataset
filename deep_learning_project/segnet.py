from keras.models import *
from keras.layers import *

from .model_utils import get_segmentation_model
from .model_utils import *
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .basic_model import vanilla_encoder
from .resnet50 import get_resnet50_encoder


def segnet_decoder(f, n_classes, n_up=3):
    assert n_up >= 2

    o = f
    o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format='channels_last'))(o)
    o = (BatchNormalization())(o)

    for _ in range(n_up-2):
        o = (UpSampling2D((2, 2), data_format='channels_last'))(o)
        o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
        o = (Conv2D(128, (3, 3), padding='valid', data_format='channels_last'))(o)
        o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format='channels_last'))(o)
    o = (ZeroPadding2D((1, 1), data_format='channels_last'))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format='channels_last', name='seg_feats'))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same', data_format='channels_last')(o)

    return o

def base_segnet(n_classes, encoder, input_height=416, input_width=608, encoder_level=3, channels=3):
    img_input, levels = encoder(input_height=input_height, input_width=input_width, channels=channels)
    feat = levels[encoder_level]
    o = segnet_decoder(feat, n_classes, n_up=3)
    model = get_segmentation_model(img_input, o)
    return model

def segnet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):
    model = base_segnet(n_classes, vanilla_encoder,  input_height=input_height, input_width=input_width, encoder_level=encoder_level, channels=channels)
    model.model_name = 'segnet'
    return model

def vgg_segnet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):
    model = base_segnet(n_classes, get_vgg_encoder,  input_height=input_height, input_width=input_width, encoder_level=encoder_level, channels=channels)
    model.model_name = 'vgg_segnet'
    return model

def resnet50_segnet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):
    model = base_segnet(n_classes, get_resnet50_encoder, input_height=input_height, input_width=input_width, encoder_level=encoder_level, channels=channels)
    model.model_name = 'resnet50_segnet'
    return model

def mobilenet_segnet(n_classes, input_height=224, input_width=224, encoder_level=3, channels=3):
    model = base_segnet(n_classes, get_mobilenet_encoder, input_height=input_height, input_width=input_width,
                        encoder_level=encoder_level, channels=channels)

    model.model_name = 'mobilenet_segnet'
    return model

if __name__ == '__main__':
    m = vgg_segnet(101)
    m = segnet(101)
    m = mobilenet_segnet(101)
