# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 16:54:40 2022

@author: 103077
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

width, height, n_frames, fps = 320, 240, 10, 1


def cv2_imshow(a, **kwargs):
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    # https://matplotlib.org/stable/gallery/showcase/mandelbrot.html#sphx-glr-gallery-showcase-mandelbrot-py
    dpi = 72
    width, height = a.shape[1], a.shape[0]
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)  # Create new figure
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)  # Add axes to figure
    ax.imshow(a, **kwargs)
    plt.axis('off')
    plt.show(block=False)  # Show image without "blocking"        


def make_image(i):
    """ Build synthetic BGR image for testing """
    p = width//60
    im = np.full((height, width, 3), 60, np.uint8)
    cv2.putText(im, str(i+1), (width//2-p*10*len(str(i+1)), height//2+p*10), cv2.FONT_HERSHEY_DUPLEX, p, (255, 30, 30), p*2)  # Blue number
    return im


# Show synthetic images in a loop
for i in range(n_frames):
    a = make_image(i)
    cv2_imshow(a)
    plt.pause(1/fps)

    # https://stackoverflow.com/a/59736741/4926757    
    clear_output(wait=False)