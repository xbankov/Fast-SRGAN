#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--image_dir', default="/home/xbankov/ahisto/ahisto-150dpi-images/", type=str,
                    help='Directory where images are kept.')
parser.add_argument('--output_dir', default="/home/xbankov/ahisto-SRGAN-images/", type=str,
                    help='Directory where to output high res images.')
parser.add_argument('--model', default="generator", type=str, help='Relative pathname to generator.h5 file.')


def main():
    args = parser.parse_args()

    # Get all image path
    p = Path(args.image_dir)
    image_paths = [img for img in p.glob("**/*") if str(img)[-4:] in ['.jpg', '.png', '.tif']]
    test = pd.read_csv("input-human-judgements-upscaled_high-confidence_filenames", sep="\t", header=None, usecols=[0])
    test = test.iloc[:, 0].str.slice(2, -4).values
    image_paths = [str(img) for img in image_paths if img.parent.stem + '/' + img.stem in test]

    # Change model input shape to accept all size inputs
    model = keras.models.load_model(f'models/{args.model}_generator.h5')
    inputs = keras.Input((None, None, 3))
    output = model(inputs)
    model = keras.models.Model(inputs, output)
    # # Create a new input layer to replace the (None,None,None,3) input layer :

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
        out = Path(args.output_dir) / Path(image_path).parent.stem
        out.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out / Path(image_path).name), sr)


if __name__ == '__main__':
    main()
