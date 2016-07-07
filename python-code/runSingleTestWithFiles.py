# runSingleTestWithFiles.py ---
#
# Filename: runSingleTestWithFiles.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Thu Jan 15 16:29:17 2015 (+0100)
# Version:
# Package-Requires: Theano
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

import sys
import time

import numpy

from Utils.custom_types import paramStruct, pathConfig
from Utils.evaluate import testModelNew
from Utils.kp_tools import (IDX_ANGLE, loadKpListFromTxt, saveKpListToTxt,
                            update_affine)

if __name__ == '__main__':
    """ Main routine """

    if len(sys.argv) != 5:
        raise RuntimeError(
            "USAGE: python runSingleTestWithFiles.py "
            "<image_file> <kp_file> <config_file> <output_file>")

    image_file_name = sys.argv[1]
    kp_file_name = sys.argv[2]
    config_file = sys.argv[3]
    output_file = sys.argv[4]

    # ------------------------------------------
    # Setup and load parameters
    param = paramStruct()
    param.loadParam(config_file)

    if 'sModelEpoch' in param.__dict__.keys():
        model_epoch = param.sModelEpoch
    else:
        model_epoch = ""

    # ------------------------------------------
    # Setup path
    pathconf = pathConfig()
    # We only need the result dir setup
    pathconf.result = config_file.replace(".config", "/")

    # ------------------------------------------
    # Run Evaluate
    start_time = time.clock()
    eval_res = testModelNew(
        image_file_name, kp_file_name, pathconf, param, model_epoch)
    end_time = time.clock()
    print("Time taken to compute for image {} (including compile time)"
          " is {} seconds".format(
              image_file_name, end_time - start_time))

    # ------------------------------------------
    # Save Results
    est_angles = numpy.asarray(eval_res[0])
    kp_list = loadKpListFromTxt(kp_file_name)
    for idxKp in xrange(len(kp_list)):
        # Update Angle
        kp_list[idxKp][IDX_ANGLE] = est_angles[idxKp] % 360.0
        # Update Affine Accordingly
        kp_list[idxKp] = update_affine(kp_list[idxKp])

    saveKpListToTxt(kp_list, kp_file_name, output_file)

#
# runSingleTestWithFiles.py ends here
