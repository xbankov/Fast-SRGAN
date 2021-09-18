#!/usr/bin/env python

import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
from tqdm.auto import tqdm

from dataloader import DataLoader, augmentation_fn
from main import pretrain, train
from model import FastSRGAN
import tensorflow as tf
import pandas as pd
import numpy as np

parser = ArgumentParser()
parser.add_argument('--train_image_dir', default="/nlp/projekty/ahisto/ahisto-superresolution/cdb/", type=str,
                    help='Path to high resolution image directory.')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training.')
parser.add_argument('--pretrain_epochs', default=1, type=int,
                    help='Number of epochs for training SRResNet without discriminator.')
parser.add_argument('--train_epochs', default=10, type=int, help='Number of epochs for training.')

parser.add_argument('--hr_size', default=384, type=int, help='Low resolution input size.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimizers.')
parser.add_argument('--save_iter', default=10, type=int,
                    help='The number of iterations to save the tensorboard summaries and models.')
parser.add_argument('--model', default='generator', type=str, help='Name of the model to be saved.')

parser.add_argument('--infer_image_dir',
                    default="/nlp/projekty/ahisto/ahisto-superresolution/ahisto-150dpi-images/",
                    type=str,
                    help='Directory where images are kept.')

parser.add_argument('--rotate_angle', default=None, type=float)
parser.add_argument('--salt_and_pepper_amount', default=None, type=float)
parser.add_argument('--salt_and_pepper_ratio', default=None, type=float)
parser.add_argument('--brightness_adjustment', default=None, type=int)
parser.add_argument('--jpeg_quality', default=None, type=int)


def main():
    # Parse the CLI arguments.
    args = parser.parse_args()

    # create directory for saving trained models.
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists(f'models/{args.model}.h5'):

        # Create the tensorflow dataset.
        logging.info("Preparing dataset")
        ds = DataLoader(args.train_image_dir, args.hr_size).dataset(args.batch_size)

        logging.info("Initializing GAN")
        # Initialize the GAN object.
        gan = FastSRGAN(args)

        # Define the directory for saving pre-training loss tensorboard summary.
        pretrain_summary_writer = tf.summary.create_file_writer(f'logs/pretrain/{args.model}')

        aug_fn = augmentation_fn(args)
        logging.info("Pretraining ...")
        # Run SRResNet pre-training.
        log_iter = args.save_iter
        for _ in range(args.pretrain_epochs):
            pretrain(gan, ds, log_iter, pretrain_summary_writer, args.model, aug_fn=aug_fn)

        # Define the directory for saving the SRGAN training tensorboard summary.
        train_summary_writer = tf.summary.create_file_writer(f'logs/train/{args.model}')

        logging.info("Training")
        # Run training.
        for _ in range(args.train_epochs):
            train(gan, ds, args.save_iter, train_summary_writer, args.model, aug_fn=aug_fn)
        logging.info("Done")

    p = Path(args.infer_image_dir)
    image_paths = [img for img in p.glob("**/*") if str(img)[-4:] in ['.jpg', '.png', '.tif']]
    test = pd.read_csv("input-human-judgements-upscaled_high-confidence_filenames", sep="\t", header=None, usecols=[0])
    test = test.iloc[:, 0].str.slice(2, -4).values
    image_paths = [str(img) for img in image_paths if img.parent.stem + '/' + img.stem in test]

    # Change model input shape to accept all size inputs

    model = tf.keras.models.load_model(f'models/{args.model}.h5')
    inputs = tf.keras.Input((None, None, 3))
    output = model(inputs)
    model = tf.keras.models.Model(inputs, output)
    out_dir = Path(f'/nlp/projekty/ahisto/ahisto-superresolution/ahisto-{args.model}-images')

    # Loop over all images
    for image_path in tqdm(image_paths):
        # Read image
        low_res = cv2.imread(image_path, 1)

        # Convert to RGB (opencv uses BGR as default)
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)

        # Rescale to 0-1.
        low_res = low_res / 255.0

        # Get super resolution image
        sr = model.predict(np.expand_dims(low_res, axis=0))[0]

        # Rescale values in range 0-255
        sr = (((sr + 1) / 2.) * 255).astype(np.uint8)

        # Convert back to BGR for opencv
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

        # Save the results:

        out = out_dir / Path(image_path).parent.stem
        out.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out / Path(image_path).name), sr)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    main()
