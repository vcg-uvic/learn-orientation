# data_tools.py ---
#
# Filename: data_tools.py
# Description: Python Module for Loading and Preparing Data
# Author: Kwang
# Maintainer:
# Created: Fri Jan 16 11:45:26 2015 (+0100)
# Version:
# Package-Requires: ()
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
# Copyright (C), EPFL Computer Vision Lab.

# Code:

from __future__ import print_function

import time
from copy import deepcopy

import numpy as np
import theano

import cv2
from Utils.imageproc import preProcessPatch
from Utils.kp_tools import (IDX_A0, IDX_A1, IDX_A2, IDX_A3, IDX_ANGLE, IDX_X,
                            IDX_Y, loadKpListFromTxt, update_affine)


def getSinglePatchRawData(img, resized_img, kp, param, random_rotation):
    """Funtion for retrieving raw region nearby the keypoint

    Parameters
    ----------

    img: original image
        Original full image to retrive data from. Should be the raw image in
        its original form

    resized_img: the original image, but resized to the patch size
        This image is used in case we want to use the full image as an
        additional feature. For example trying the global orientation.

    kp: the keypoint
        The keypoint structure of the target point. This is not a list!

    param: struct
        The parameter structure

    random_rotation: float, (degrees)
        The rotation to be applied when extracting the patch. Note that this
        completely replaces at which angle we are extracting the patch. Since
        we extract patches upright in the cvpr16 version, it is good enough
        that we put the desired alterations on the rotations here.

    """

    assert img.dtype == 'float32'

    scaleMultiplier = param.fRatioScale
    # scaleMultiplier /= np.float32(param.nPatchSize)*0.5

    # ------------------------------------------------------------------------
    # Patch extraction using Affine Information!

    # Recover the upright A from a,b,c
    upright_kp = kp.copy()
    upright_kp[IDX_ANGLE] = random_rotation
    upright_kp = update_affine(upright_kp)

    # The Affine Matrix (with orientation)
    UprightA = np.asarray([[upright_kp[IDX_A0], upright_kp[IDX_A1]],
                           [upright_kp[IDX_A2], upright_kp[IDX_A3]]])

    # Rescale the uprightA according to parameters (looking at larger region!)
    UprightA *= scaleMultiplier

    # Add bias in the translation vector so that we get from -1 to 1
    t = np.asarray([[upright_kp[IDX_X], upright_kp[IDX_Y]]]).T + \
        np.dot(UprightA, np.asarray([[-1], [-1]]))

    # Transform in OpenCV representation

    # scaled uprightA so that we can use cv2.warpAffine (0~patchsize)
    M = np.concatenate(
        (UprightA / (float(param.nPatchSize) * 0.5), t),
        axis=1
    )

    # TODO:THIS PART MIGHT BE SLOW!!!
    patch_data = cv2.warpAffine(
        img, M, (param.nPatchSize, param.nPatchSize),
        flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101)

    # ------------------------------------------------------------------------
    # expand dim since opencv trashes unused dim
    if not len(patch_data.shape) == 3:
        patch_data = np.expand_dims(patch_data, axis=2)

    patch_data = preProcessPatch(patch_data, resized_img, param)

    # expand dim since opencv trashes unused dim
    if not len(patch_data.shape) == 3:
        patch_data = np.expand_dims(patch_data, axis=2)

    return np.transpose(patch_data, axes=[2, 0, 1]).astype(
        theano.config.floatX
    )


def loadPatchData(img_file_name, kp_file_name, param, bVerboseTime=False):
    """ Function for loading the data into the format our network wants """

    if bVerboseTime:
        start_time = time.clock()

    # Load image
    img = cv2.imread(img_file_name)
    # keep number of dimensions to 3
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    # change image into float32
    img = img.astype('float32')

    if bVerboseTime:
        end_time = time.clock()
        print("Time taken to load image is {} seconds".format(
            end_time - start_time))

    # We don't measure the resized_img as we don't use it
    resized_img = cv2.resize(img, (param.nPatchSize, param.nPatchSize))

    if bVerboseTime:
        start_time = time.clock()

    # Load keypoints
    kp_list = loadKpListFromTxt(kp_file_name)

    if bVerboseTime:
        end_time = time.clock()
        print("Time taken to load kp is {} kps is {} seconds".format(
            len(kp_list), end_time - start_time))

    # Prepare params for pyramid learning case
    param_half_scale = deepcopy(param)
    param_half_scale.fRatioScale *= 0.5
    param_quarter_scale = deepcopy(param)
    param_quarter_scale.fRatioScale *= 0.25

    if bVerboseTime:
        start_time = time.clock()

    # For each keypoint load data
    raw_data_list = []
    for kp in kp_list:
        # pass zero as random rotation
        raw_data_list += [getSinglePatchRawData(
            img, resized_img, kp, param, 0)]

    if not param.bNormalizePatch:
        raw_data_list = [raw_data / 255. for raw_data in raw_data_list]

    raw_data = np.asarray(raw_data_list)

    if bVerboseTime:
        end_time = time.clock()
        print("Time taken to load patches for {} kps is {} seconds".format(
            len(kp_list), end_time - start_time))

    return raw_data


#
# data_tools.py ends here
