import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import array_ops, math_ops


class DataLoader:
    """Data Loader for the SR GAN, that prepares a tf data object for training."""

    def __init__(self, image_dir, hr_image_size):
        """
        Initializes the dataloader.
        Args:
            image_dir: The path to the directory containing high resolution images.
            hr_image_size: Integer, the crop size of the images to train on (High
                           resolution images will be cropped to this width and height).
        Returns:
            The dataloader object.
        """
        if image_dir.split('.')[-1] == 'txt':
            with open(image_dir, mode='r') as f:
                self.image_paths = [os.path.join('/ahisto/', f[:-1]) for f in f.readlines() if f[:-1].endswith('.png')]
        else:
            self.image_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.image_size = hr_image_size

    def _parse_image(self, image_path):
        """
        Function that loads the images given the path.
        Args:
            image_path: Path to an image file.
        Returns:
            image: A tf tensor of the loaded image.
        """

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Check if image is large enough
        if tf.keras.backend.image_data_format() == 'channels_last':
            shape = array_ops.shape(image)[:2]
        else:
            shape = array_ops.shape(image)[1:]
        cond = math_ops.reduce_all(shape >= tf.constant(self.image_size))

        image = tf.cond(cond, lambda: tf.identity(image),
                        lambda: tf.image.resize(image, [self.image_size, self.image_size]))

        return image

    def _random_crop(self, image):
        """
        Function that crops the image according a defined width
        and height.
        Args:
            image: A tf tensor of an image.
        Returns:
            image: A tf tensor of containing the cropped image.
        """

        image = tf.image.random_crop(image, [self.image_size, self.image_size, 3])

        return image

    @staticmethod
    def _to_grayscale(image):
        """
        Function that convert rgb to grayscale.
        Args:
            image: A tf tensor of an rgb image.
        Returns:
            image: A tf tensor of containing the grayscale image.
        """

        image = tf.image.rgb_to_grayscale(image)
        return image

    def _high_low_res_pairs(self, high_res):
        """
        Function that generates a low resolution image given the 
        high resolution image. The downsampling factor is 4x.
        Args:
            high_res: A tf tensor of the high res image.
        Returns:
            low_res: A tf tensor of the low res image.
            high_res: A tf tensor of the high res image.
        """

        low_res = tf.image.resize(high_res,
                                  [self.image_size // 4, self.image_size // 4],
                                  method='bicubic')

        return low_res, high_res

    @staticmethod
    def _rescale(low_res, high_res):
        """
        Function that rescales the pixel values to the -1 to 1 range.
        For use with the generator output tanh function.
        Args:
            low_res: The tf tensor of the low res image.
            high_res: The tf tensor of the high res image.
        Returns:
            low_res: The tf tensor of the low res image, rescaled.
            high_res: the tf tensor of the high res image, rescaled.
        """
        high_res = high_res * 2.0 - 1.0

        return low_res, high_res

    def dataset(self, batch_size):
        """
        Returns a tf dataset object with specified mappings.
        Args:
            batch_size: Int, The number of elements in a batch returned by the dataset.
        Returns:
            dataset: A tf dataset object.
        """

        # Generate tf dataset from high res image paths.
        dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)

        # Read the images
        dataset = dataset.map(self._parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Crop out a piece for training
        dataset = dataset.map(self._random_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Grayscale
        # dataset = dataset.map(self._to_grayscale, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Generate low resolution by down-sampling crop.
        dataset = dataset.map(self._high_low_res_pairs, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Rescale the values in the input
        dataset = dataset.map(self._rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Batch the input, drop remainder to get a defined batch size.
        # Prefetch the data for optimal GPU utilization.
        dataset = dataset.shuffle(30).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


def rotate(image, angle):
    clockwise = tf.random.uniform([], seed=123) > 0.5
    angle = tf.random.uniform([], 0, 0 + angle, seed=123) if clockwise else tf.random.uniform([], 360 - angle, 360,
                                                                                              seed=123)
    image = tfa.image.rotate(image, np.radians(angle), interpolation="bilinear", fill_mode='CONSTANT', fill_value=1.0)
    return image


def salt_and_pepper(image, p, q):
    width = image.shape[0]
    height = image.shape[1]

    flipped = np.random.choice([True, False], size=(width, height), p=[p, 1 - p])

    salted = np.random.choice([True, False], size=(width, height), p=[q, 1 - q])
    peppered = ~salted

    image = np.asarray(image).copy()
    image[flipped & salted] = (1.0, 1.0, 1.0)
    image[flipped & peppered] = (0.0, 0.0, 0.0)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def brightness(image, brightness_adjustment):
    plus = tf.random.uniform([], seed=123) > 0.5
    if plus:
        adjust_value = tf.random.uniform([], 0, 0 + brightness_adjustment, seed=123)
    else:
        adjust_value = tf.random.uniform([], 0, 0 - brightness_adjustment, seed=123)
    return tf.image.adjust_brightness(image, adjust_value)


def add_jpeg_noise(image, lowest_quality):
    quality = tf.random.uniform([], 100 - lowest_quality, 100, seed=123, dtype=tf.dtypes.int32)
    return tf.image.adjust_jpeg_quality(image, jpeg_quality=quality)


def augmentation_fn(args):
    def f(image):
        if args.rotate_angle and 0 < args.rotate_angle < 360:
            image = rotate(image, args.rotate_angle)
        if args.salt_and_pepper_amount and args.salt_and_pepper_ratio:
            image = salt_and_pepper(image, args.salt_and_pepper_amount, args.salt_and_pepper_ratio)
        if args.brightness_adjustment:
            image = brightness(image, args.brightness_adjustment)
        if args.jpeg_quality:
            image = add_jpeg_noise(image, args.jpeg_quality)
        return image

    return f
