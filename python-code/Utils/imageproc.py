# imageproc.py ---
#
# Filename: imageproc.py
# Description: Python Module with Image Processing Functions
#              (including descriptor fields)
# Author: Kwang
# Maintainer:
# Created: Fri Jan 16 17:50:08 2015 (+0100)
# Version:
# Package-Requires: ()
# Last-Updated: Mon Jul  4 05:14:50 2016 (-0700)
#           By: Kwang Moo Yi
#     Update #: 166
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
#
# Copyright (C), EPFL Computer Vision Lab.
#
#

# Code:

from copy import deepcopy

import numpy
import theano

import cv2


def preProcessPatch(img, orig_img, param):
    """ Per Patch Pre Processing """

    pre_pre_img = deepcopy(img)

    pre_proc_list = param.sDataType.split('_')

    pre_proc_img_list = []
    for pre_proc in pre_proc_list:
        # pre-process image according to parameter. Will raise flag if
        # unknown option is given
        pre_proc_img_list += [{
            # RGB raw image
            'RGB': lambda pre_pre_img, param: getRGB(
                pre_pre_img, param),
            # Gray scale image
            'Gray': lambda pre_pre_img, param: getGray(
                pre_pre_img, param)
        }[pre_proc](pre_pre_img, param)]

    # concatenate in the third dim
    ret_img = numpy.concatenate(pre_proc_img_list, axis=2)

    if not len(ret_img.shape) == 3:
        ret_img.shape = [ret_img.shape[0], ret_img.shape[1], 1]

    return ret_img


def getRGB(img, param):

    ret_img = img.copy().astype(theano.config.floatX)

    if param.bNormalizePatch:
        ret_img = normalize(ret_img)

    return ret_img


def getGray(img, param):
    ret_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # expand dim since opencv trashes unused dim
    ret_img = numpy.expand_dims(ret_img, axis=2).astype(theano.config.floatX)

    if param.bNormalizePatch:
        ret_img = normalize(ret_img)

    return ret_img


def normalize(img):
    """ Normalize image using mean and std """

    ret_img = numpy.zeros_like(img).astype(theano.config.floatX)

    for idxC in xrange(img.shape[2]):
        [mu, sigma] = cv2.meanStdDev(img[:, :, idxC])
        if sigma == 0:
            ret_img[:, :, idxC] = (img[:, :, idxC] - mu)
        else:
            ret_img[:, :, idxC] = (img[:, :, idxC] - mu) / sigma

    return ret_img


#
# imageproc.py ends here
