#!/usr/bin/env python

import logging
import os
from argparse import ArgumentParser

import tensorflow as tf
from tqdm.auto import tqdm

from dataloader import DataLoader
from model import FastSRGAN

parser = ArgumentParser()
parser.add_argument('--image_dir', default="/nlp/projekty/ahisto/ahisto-superresolution/cdb/", type=str,
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


@tf.function
def pretrain_step(model, x, y):
    """
    Single step of generator pre-training.
    Args:
        model: A model object with a tf keras compiled generator.
        x: The low resolution image tensor.
        y: The high resolution image tensor.
    """
    with tf.GradientTape() as tape:
        fake_hr = model.generator(x)
        loss_mse = tf.keras.losses.MeanSquaredError()(y, fake_hr)

    grads = tape.gradient(loss_mse, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(zip(grads, model.generator.trainable_variables))

    return loss_mse


def pretrain(model, dataset, log_iter, writer, model_name, aug_fn):
    """Function that pretrains the generator slightly, to avoid local minima.
    Args:
        model: The keras model to train.
        dataset: A tf dataset object of low and high res images to pretrain over.
        log_iter: Log each log_iter iterations
        writer: A summary writer object.
        model_name: A name of the model_name to save
        aug_fn: function to use as a image x augmentation
    Returns:
        None
    """
    with writer.as_default():
        for x, y in tqdm(dataset):
            if aug_fn is not None:
                x = tf.map_fn(aug_fn, x)
            loss = pretrain_step(model, x, y)
            if model.pretrain_iterations % log_iter == 0:
                tf.summary.scalar('MSE Loss', loss, step=tf.cast(model.pretrain_iterations, tf.int64))
                tf.summary.image('Low Res', tf.cast(255 * x, tf.uint8), step=model.pretrain_iterations)
                tf.summary.image('High Res', tf.cast(255 * (y + 1.0) / 2.0, tf.uint8), step=model.pretrain_iterations)
                tf.summary.image('Generated', tf.cast(255 * (model.generator.predict(x) + 1.0) / 2.0, tf.uint8),
                                 step=model.pretrain_iterations)
                model.generator.save(f'models/{model_name}.h5')
                writer.flush()
            model.pretrain_iterations += 1


@tf.function
def train_step(model, x, y):
    """Single train step function for the SRGAN.
    Args:
        model: An object that contains a tf keras compiled discriminator model.
        x: The low resolution input image.
        y: The desired high resolution output image.

    Returns:
        d_loss: The mean loss of the discriminator.
    """
    # Label smoothing for better gradient flow
    valid = tf.ones((x.shape[0],) + model.disc_patch)
    fake = tf.zeros((x.shape[0],) + model.disc_patch)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # From low res. image generate high res. version
        fake_hr = model.generator(x)

        # Train the discriminators (original images = real / generated = Fake)
        valid_prediction = model.discriminator(y)
        fake_prediction = model.discriminator(fake_hr)

        # Generator loss
        content_loss = model.content_loss(y, fake_hr)
        adv_loss = 1e-3 * tf.keras.losses.BinaryCrossentropy()(valid, fake_prediction)
        mse_loss = tf.keras.losses.MeanSquaredError()(y, fake_hr)
        perceptual_loss = content_loss + adv_loss + mse_loss

        # Discriminator loss
        valid_loss = tf.keras.losses.BinaryCrossentropy()(valid, valid_prediction)
        fake_loss = tf.keras.losses.BinaryCrossentropy()(fake, fake_prediction)
        d_loss = tf.add(valid_loss, fake_loss)

    # Backprop on Generator
    gen_grads = gen_tape.gradient(perceptual_loss, model.generator.trainable_variables)
    model.gen_optimizer.apply_gradients(zip(gen_grads, model.generator.trainable_variables))

    # Backprop on Discriminator
    disc_grads = disc_tape.gradient(d_loss, model.discriminator.trainable_variables)
    model.disc_optimizer.apply_gradients(zip(disc_grads, model.discriminator.trainable_variables))

    return d_loss, adv_loss, content_loss, mse_loss


def train(model, dataset, log_iter, writer, model_name, aug_fn):
    """
    Function that defines a single training step for the SR-GAN.
    Args:
        model: An object that contains tf keras compiled generator and
               discriminator models.
        dataset: A tf data object that contains low and high res images.
        log_iter: Number of iterations after which to add logs in 
                  tensorboard.
        writer: Summary writer
        model_name: A name of the model_name to save
        aug_fn: function to use as a image x augmentation
    """
    with writer.as_default():
        # Iterate over dataset
        for x, y in tqdm(dataset):
            if aug_fn is not None:
                x = tf.map_fn(aug_fn, x)
            disc_loss, adv_loss, content_loss, mse_loss = train_step(model, x, y)
            # Log tensorboard summaries if log iteration is reached.
            if model.iterations % log_iter == 0:
                tf.summary.scalar('Adversarial Loss', adv_loss, step=model.iterations)
                tf.summary.scalar('Content Loss', content_loss, step=model.iterations)
                tf.summary.scalar('MSE Loss', mse_loss, step=model.iterations)
                tf.summary.scalar('Discriminator Loss', disc_loss, step=model.iterations)
                tf.summary.image('Low Res', tf.cast(255 * x, tf.uint8), step=model.iterations)
                tf.summary.image('High Res', tf.cast(255 * (y + 1.0) / 2.0, tf.uint8), step=model.iterations)
                tf.summary.image('Generated', tf.cast(255 * (model.generator.predict(x) + 1.0) / 2.0, tf.uint8),
                                 step=model.iterations)

                model.generator.save(f'models/{model_name}.h5')
                model.discriminator.save(f'models/{model_name}_discriminator.h5')
                writer.flush()
            model.iterations += 1


def main():
    # Parse the CLI arguments.
    args = parser.parse_args()

    # create directory for saving trained models.
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists(f'logs/{args.model}'):
        os.makedirs(f'logs/{args.model}')

    # Create the tensorflow dataset.
    logging.info("Preparing dataset")
    ds = DataLoader(args.image_dir, args.hr_size).dataset(args.batch_size)

    logging.info("Initializing GAN")
    # Initialize the GAN object.
    gan = FastSRGAN(args)

    # Define the directory for saving pre-training loss tensorboard summary.
    pretrain_summary_writer = tf.summary.create_file_writer(f'logs/pretrain/{args.model}')

    logging.info("Pretraining ...")
    # Run SRResNet pre-training.
    log_iter = args.save_iter

    for _ in range(args.pretrain_epochs):
        pretrain(gan, ds, log_iter, pretrain_summary_writer, args.model, aug_fn=None)

    # Define the directory for saving the SRGAN training tensorboard summary.
    train_summary_writer = tf.summary.create_file_writer(f'logs/train/{args.model}')

    logging.info("Training")
    # Run training.
    for _ in range(args.train_epochs):
        train(gan, ds, args.save_iter, train_summary_writer, args.model, aug_fn=None)
    logging.info("Done")


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    main()
