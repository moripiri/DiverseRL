from typing import Tuple, Union

import numpy as np
import torch


def center_crop(images: torch.Tensor, output_size: Union[int, Tuple], data_format: str = 'channels_first') -> torch.Tensor:
    '''
    Center crop an image or images in torch tensor

    :param images: images in numpy array
    :param output_size: width and height of center cropped image size
    :param data_format: format of the input image. 'channels_first', or 'channels_last'

    :return: center cropped images in output size
    '''

    assert images.ndim in (3, 4), "Image type must be numpy array, and its dimension must be 3 or 4"
    assert data_format in ['channels_first', 'channels_last'], "data_format must be either channels_first or channels_last"

    original_ndim = images.ndim
    if original_ndim == 3:
        images = torch.unsqueeze(images, dim=0)

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if data_format == 'channels_first':
        b, c, h, w = images.shape
    else:
        b, h, w, c = images.shape

    assert h >= output_size[0] and w >= output_size[1]

    new_h, new_w = output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    if data_format == 'channels_first':
        images = images[:, :, top:top + new_h, left:left + new_w]
    else:

        images = images[:, top: top + new_h, left:left + new_w, :]

    if original_ndim == 3:
        images = images[0]

    return images


def random_crop(images: torch.Tensor, output_size: Union[int, Tuple], data_format: str = 'channels_first') -> torch.Tensor:
    """
    Randomly crop an image or stacked images in torch tensor

    :param images: image in torch tensor
    :param output_size: width and height of randomly cropped image size
    :param data_format: format of the input image. 'channels_first', or 'channels_last'
    :return:
    """
    assert images.ndim in (3, 4), "Number of image dimension must be 3 or 4"
    assert data_format in ['channels_first', 'channels_last']

    original_ndim = images.ndim
    if original_ndim == 3:
        images = torch.unsqueeze(images, dim=0)

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if data_format == 'channels_first':
        b, c, h, w = images.shape
    else:
        b, h, w, c = images.shape

    assert h >= output_size[0] and w >= output_size[1]

    crop_max_h = h - output_size[0] + 1
    crop_max_w = w - output_size[1] + 1

    w1 = np.random.randint(0, crop_max_w, b)
    h1 = np.random.randint(0, crop_max_h, b)

    if data_format == 'channels_first':
        cropped = torch.empty((images.shape[0], images.shape[1], output_size[0], output_size[1]), dtype=images.dtype)
        for i, (images, w11, h11) in enumerate(zip(images, w1, h1)):
            cropped[i] = images[:, h11:h11 + output_size[0], w11: w11 + output_size[1]]

    else:
        cropped = torch.empty((images.shape[0], output_size[0], output_size[1], images.shape[-1]), dtype=images.dtype)
        for i, (images, w11, h11) in enumerate(zip(images, w1, h1)):
            cropped[i] = images[h11:h11 + output_size[0], w11: w11 + output_size[1], :]

    if original_ndim == 3:
        cropped = cropped[0]

    return cropped
