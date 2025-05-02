import cv2
import os
from psd_tools import PSDImage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize


def fill_contours(layer, is_disc, subject):
    # 2) to OpenCV BGR
    img = cv2.cvtColor(np.array(layer), cv2.COLOR_RGBA2BGR)

    # 3) isolate the “red dots” (assumes they’re pure red)
    #    you may need to tweak these thresholds if your dots are anti-aliased!
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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # close = dilate then erode → fills gaps
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # optionally, thin the result back to a 1-pixel wide skeleton
    skel = skeletonize(closed//255)       # skeletonize wants a 0/1 image
    skel = (skel.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(
        skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2) make a blank canvas
    filled = np.zeros_like(closed)

    # 3) draw each contour filled
    cv2.drawContours(filled, contours, contourIdx=-1,
                     color=255, thickness=cv2.FILLED)

    file_name = f'{subject}-disc.png' if is_disc else f'{subject}-cup.png'
    cv2.imwrite(file_name, filled)


psd = PSDImage.open("Subject1-cd1-2.psd")
print(len(psd._layers))
viewport = [0, 0, psd[0].size[0], psd[0].size[1]]
for idx, layer in enumerate(psd):
    if idx > 0:
        fill_contours(layer.composite(viewport=viewport),
                    is_disc=idx == 1, subject="Subject1-cd1")
    else:
        layer.composite(viewport=viewport).save('Subject1-cd1-2.png')
