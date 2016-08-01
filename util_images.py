"""
This module contains utility functions involving reading and writing image files
"""

from __future__ import division
from __future__ import print_function

__author__ = "Ankit Laddha <aladdha@andrew.cmu.edu>"

from PIL import Image as imlib
import numpy as np
from skimage.color.colorlabel import label2rgb


def load_image_pil(fname):
    """
    Load an image at path "fname" using PIL Library

    Parameters
    ----------
    fname : string
        Path of the image to read

    Returns
    -------
    img : (M,N,channels)
        An numpy array with the image data
    """
    im = imlib.open(fname)
    img = np.array(im.getdata(), np.uint8)
    channels = 1 # by default we only have one channel in the image
    if hasattr(im, 'layers'):
        channels = 3

    img = img.reshape(im.height, im.width, channels)

    return img


def overlay_labels(labels, image, colormap, alpha):
    """
    Overlay the labels on the image using the colormap
    :param

    labels : (M,N)
        A numpy array with integer entries which contains the labels.
        Labels must start from 0
    image : (M,N,3)
        A numpy array of type uint8 which contains the image data
    colormap : (N,3)
        A list which contains the colormaps. The entries are of type uint8
    alpha : float
        The opacity of the labelmap
    bglabel : int
        label of background class.
        Pixel labels as bg won't be overlayed with color

    Returns
    -------
    overlayedImg : (M,N,3)
        A numpy array of type uint8
    """

    colors = label2rgb(labels, colors=colormap)
    colors = colors.astype(np.float32)
    image = image.astype(np.float32)
    overlayedImg = colors*alpha + (1-alpha)*image
    overlayedImg = overlayedImg.astype(np.uint8)

    return overlayedImg