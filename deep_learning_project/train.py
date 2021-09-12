import json
import os
import six
from keras.losses import categorical_crossentropy
import tensorflow as tf
import glob
import sys

from .data_loader import image_segmentation_generator


def masked_categorical_crossentropy(gt, pr):
    mask = 1 - gt[:, :, 0]
    return categorical_crossentropy(gt, pr) * mask

def train(model, train_images, train_annotations, input_height=None, input_width=None, n_classes=None, verify_dataset=True,
          epochs=5, batch_size=2, steps_per_epoch=512, gen_use_multiprocessing=False, ignore_zero_class=False,
          optimizer_name='adam', other_inputs_paths=None, read_image_type=1):

    from .all_models import model_names

    if isinstance(model, six.string_types):
        assert (n_classes is not None), 'Please provide the n_classes'
        if (input_height is not None) and (input_width is not None):
            model = model_names[model](n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_names[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if optimizer_name is not None:
        if ignore_zero_class:
            loss_k = masked_categorical_crossentropy
        else:
            loss_k = 'categorical_crossentropy'

        model.compile(loss=loss_k, optimizer=optimizer_name, metrics=['accuracy'])
        model.summary()

    initial_epoch = 0

    train_gen = image_segmentation_generator(train_images, train_annotations, batch_size, n_classes, input_height, input_width,
                                             output_height, output_width, other_inputs_paths=other_inputs_paths, read_image_type=read_image_type)

    model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=initial_epoch)
