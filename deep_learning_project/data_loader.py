import itertools
import os
import random
import six
import numpy as np
import cv2
from collections.abc import Sequence
from tqdm import tqdm


class DataLoaderError(Exception):
    pass


acceptable_image_formats = ['.jpg', '.jpeg', '.png', '.bmp']
acceptable_segmentation_formats = ['.png', '.bmp']
data_loader_seed = 0
random.seed(data_loader_seed)
class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(5000)]

def get_image_list_from_path(images_path):
    image_files = []
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and os.path.splitext(dir_entry)[1] in acceptable_image_formats:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append(os.path.join(images_path, dir_entry))

    return image_files

def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False, other_inputs_paths=None):
    image_files = []
    segmentation_files = {}
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and os.path.splitext(dir_entry)[1] in acceptable_image_formats:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension, os.path.join(images_path, dir_entry)))

    if other_inputs_paths is not None:
        other_inputs_files = []
        for i, other_inputs_path in enumerate(other_inputs_paths):
            temp = []
            for y, dir_entry in enumerate(os.listdir(other_inputs_path)):
                if os.path.isfile(os.path.join(other_inputs_path, dir_entry)) and os.path.splitext(dir_entry)[1] in acceptable_image_formats:
                    file_name, file_extension = os.path.splitext(dir_entry)
                    temp.append((file_name, file_extension, os.path.join(other_inputs_path, dir_entry)))

            other_inputs_files.append(temp)

    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and os.path.splitext(dir_entry)[1] in acceptable_segmentation_formats:
            file_name, file_extension = os.path.splitext(dir_entry)
            full_dir_entry = os.path.join(segs_path, dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError(f'Segmentation file with filename {file_name} already exists and is ambiguous to resolve with path {full_dir_entry}. Please remove or rename the latter.')

            segmentation_files[file_name] = (file_extension, full_dir_entry)

    return_value = []
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            if other_inputs_paths is not None:
                other_inputs = []
                for file_paths in other_inputs_files:
                    success = False
                    for (other_file, _, other_full_path) in file_paths:
                        if image_file == other_file:
                            other_inputs.append(other_full_path)
                            success = True
                            break

                    if not success:
                        raise ValueError('There was no matching other input for ', image_file, ' in directory')

                return_value.append((image_full_path,segmentation_files[image_file][1], other_inputs))
            else:
                return_value.append((image_full_path, segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            raise DataLoaderError(f'No corresponding segmentation found for image {image_full_path}.')

    return return_value

def get_image_array(image_input, width, height, image_norm='sub_mean', ordering='channels_last', read_image_type=1):
    if type(image_input) is np.ndarray:
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError(f'get_image_array: path {image_input} doesn\'t exist')

        img = cv2.imread(image_input, read_image_type)
    else:
        raise DataLoaderError(f'get_image_array: Can\'t process input type {str(type(image_input))}')

    if image_norm == 'sub_and_divide':
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif image_norm == 'sub_mean':
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = np.atleast_3d(img)
        means = [103.939, 116.779, 123.68]
        for i in range(min(img.shape[2], len(means))):
            img[:, :, i] -= means[i]

        img = img[:, :, ::-1]
    elif image_norm == 'divide':
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    return img

def get_segmentation_array(image_input, n_classes, width, height, no_reshape=False, read_image_type=1):
    seg_labels = np.zeros((height, width, n_classes))
    if type(image_input) is np.ndarray:
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError(f'get_segmentation_array: path {image_input} doesn\'t exist')

        img = cv2.imread(image_input, read_image_type)
    else:
        raise DataLoaderError(f'get_segmentation_array: Can\'t process input type {str(type(image_input))}')

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]
    for c in range(n_classes):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, n_classes))

    return seg_labels

def image_segmentation_generator(images_path, segs_path, batch_size, n_classes, input_height, input_width, output_height, output_width,
                                 other_inputs_paths=None, read_image_type=cv2.IMREAD_COLOR, ignore_segs=False):

    if not ignore_segs:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path, other_inputs_paths=other_inputs_paths)
        random.shuffle(img_seg_pairs)
        zipped = itertools.cycle(img_seg_pairs)
    else:
        img_list = get_image_list_from_path(images_path)
        random.shuffle(img_list)
        img_list_gen = itertools.cycle(img_list)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            if other_inputs_paths is None:
                if ignore_segs:
                    im = next(img_list_gen)
                    seg = None
                else:
                    im, seg = next(zipped)
                    seg = cv2.imread(seg, 1)

                im = cv2.imread(im, read_image_type)
                X.append(get_image_array(im, input_width, input_height, ordering='channels_last'))
            else:
                assert ignore_segs == False , 'Not supported yet'
                im, seg, others = next(zipped)
                im = cv2.imread(im, read_image_type)
                seg = cv2.imread(seg, 1)
                oth = []
                for f in others:
                    oth.append(cv2.imread(f, read_image_type))

                ims = [im]
                ims.extend(oth)
                oth = []
                for i, image in enumerate(ims):
                    oth_im = get_image_array(image, input_width, input_height, ordering='channels_last')
                    oth.append(oth_im)

                X.append(oth)

            if not ignore_segs:
                Y.append(get_segmentation_array(seg, n_classes, output_width, output_height))

        if ignore_segs:
            yield np.array(X)
        else:
            yield np.array(X), np.array(Y)
