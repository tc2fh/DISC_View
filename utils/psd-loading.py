import cv2
import os
from psd_tools import PSDImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import sys


def fill_contours(layer, is_disc, subject):
    # Convert the layer to RGB
    img = cv2.cvtColor(np.array(layer), cv2.COLOR_RGBA2BGR)

    # isolate the red annotation if we're getting the disc mask else blue annotations
    if is_disc:
        lower = np.array([0,   0, 200])   # BGR
        upper = np.array([50,  50, 255])
        mask = cv2.inRange(img, lower, upper)
    else:
        # B ≥200, G ≥0,  R ≥0
        lower = np.array([200,   0,   0], dtype=np.uint8)
        # B ≤255, G ≤50, R ≤50
        upper = np.array([255,  50,  50], dtype=np.uint8)

        mask = cv2.inRange(img, lower, upper)

    # Set the kernel for interpolation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # close = dilate then erode → fills gaps
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # thin the result back to a 1-pixel wide skeleton
    skel = skeletonize(closed//255)       # skeletonize wants a 0-1 range image
    skel = (skel.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(
        skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # make a blank "canvas"
    filled = np.zeros_like(closed)

    # draw each contour filled
    cv2.drawContours(filled, contours, contourIdx=-1,
                     color=255, thickness=cv2.FILLED)

    # save the interpolated mask
    file_name = f'{subject}-disc.png' if is_disc else f'{subject}-cup.png'
    cv2.imwrite(file_name, filled)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python psd-loading.py <photoshop-file-path>")
        sys.exit(1)
    # open the image
    file_path = sys.argv[1]
    file_name = file_path.split('.')[0]
    psd = PSDImage.open(file_path)

    # Set the viewport boundary to get the masks to be the right size when interpolating
    viewport = [0, 0, psd[0].size[0], psd[0].size[1]]
    for idx, layer in enumerate(psd):
        if idx > 0:
            fill_contours(layer.composite(viewport=viewport),
                          is_disc=idx == 1, subject=file_name)
        else:
            layer.composite(viewport=viewport).save(f'{file_name}.png')
