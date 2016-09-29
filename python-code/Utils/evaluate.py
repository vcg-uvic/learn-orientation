# evaluate.py ---
#
# Filename: evaluate.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Fri Mar  6 11:52:42 2015 (+0100)
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
#
# Copyright (C), EPFL Computer Vision Lab.
#
#

# Code:

import time

import numpy

from Utils.data_tools import loadPatchData
from Utils.networks.buildLayers import instantiateLayers


def testModelNew(img_file_name, kp_file_name, pathconf, param, model_epoch="",
                 deterministic=False):

    if param.runType != "CVPR16":
        raise NotImplementedError(
            "Other types of networks are not supported!"
            " Modify the code and the network at your risk!"
        )

    from Utils.networks.cvpr16 import (
        SiameseOrientationLearner, SiameseOrientationLearnerConfig
    )

    print("WARNING: I am probably doing unnecessary stuff due to num_siamese")
    param.num_siamese = 3

    save_dir = pathconf.result

    # kp_file_name = pathconf.temp + "temp_kp_file.kp"
    # saveKpListToTxt(kp_list, kp_file_name)
    x_in = loadPatchData(img_file_name, kp_file_name, param, bVerboseTime=True)

    start_time = time.clock()
    # do the normalization if requested
    if param.bNormalizeInput:

        # save the mean and std used for normalization (should I de-correlate
        # the data as well?)
        input_mean = numpy.load(save_dir + "input_mean.npy")
        input_std = numpy.load(save_dir + "input_std.npy")

        x_in -= input_mean
        x_in /= input_std

    end_time = time.clock()
    print("Time taken to normalize patches for {} kps is {} seconds".format(
        len(x_in), end_time - start_time))

    # -------------------------------------------------------------------------
    # Initialize network Config
    myNetConfig = SiameseOrientationLearnerConfig()

    # -------------------------------------------------------------------------
    # Copy over all other attributes to Config
    for _key in param.__dict__.keys():
        setattr(myNetConfig, _key, getattr(param, _key))

    # -------------------------------------------------------------------------
    # Config fields which need individual attention

    # directories
    myNetConfig.save_dir = save_dir

    # dataset info
    myNetConfig.batch_size = len(x_in)
    # myNetConfig.data_dim = gt_desc_array.shape[1]
    myNetConfig.num_channel = x_in[0].shape[0]
    myNetConfig.patch_size = x_in[0].shape[2]

    # -------------------------------------------------------------------------
    # Actual instantiation and setup
    myNet = SiameseOrientationLearner(myNetConfig)

    instantiateLayers(myNet, deterministic)
    # myNet.setupSGD()
    myNet.setupDataAndCompile4Test()

    start_time = time.clock()
    res_unscaled, res_raw = myNet.runTest(x_in, model_epoch)
    end_time = time.clock()
    print("Time taken to compute {} kps"
          " with pre-loaded patch is {} seconds".format(
              len(x_in), end_time - start_time))

    return res_unscaled / numpy.pi * 180.0, res_raw

#
# evaluate.py ends here
