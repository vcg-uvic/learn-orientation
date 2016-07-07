# custom_types.py ---
#
# Filename: custom_types.py
# Description: Python Module for custom types
# Author: Kwang
# Maintainer:
# Created: Fri Jan 16 12:01:52 2015 (+0100)
# Version:
# Package-Requires: ()
# Last-Updated: Thu Jul  7 10:54:05 2016 (+0200)
#           By: Kwang Moo Yi
#     Update #: 288
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

# ------------------------------------------
# Imports
from __future__ import print_function

import numpy
from parse import parse


class pathConfig:
    """Structure for the paths"""

    dataset = None
    temp = None
    result = None
    debug = None

    train_data = None
    train_mask = None


class paramStruct:
    """ Parameter Structure  """

    def __init__(self):
        """ Initialization.

        Note: Some values are set to None to act as a sanity check

        """

        # ---------------------------------------------------------------------
        # Paramters for patch extraction
        self.nPatchSize = None           # Width and Height of the patch
        self.sDataType = None            # Image Data type for Patches
        self.bNormalizePatch = False     # Normalize single patch?
        # fRatioScale will be multiplied to the SIFT scale (negative uses fixed
        # scale)
        self.fRatioScale = None

        # ---------------------------------------------------------------------
        # Paramters for Learning
        self.sKpType = None		# the keypoint detector
        self.sDescType = None		# the descriptor type

        # ---------------------------------------------------------------------
        # Model parameters
        # Which type am I running in terms of python executable
        self.runType = None
        # Architecture, Set to None so that it crashes without config
        self.modelType = None
        self.num_siamese = None          # number of siamese clones
        self.bNormalizeInput = None      # whether to normalize the input
        self.batch_size = None           # batch size for SGD or testing

        # ---------------------------------------------------------------------
        # GHH parameters (will not have effect when using CNN)
        # number of sums in GHH, set to None so that it crashes without config
        self.GHH_numSUM = None
        # number of maxes in GHH, set to None so that it crashes without config
        self.GHH_numMAX = None

    def loadParam(self, file_name, verbose=True):

        config_file = open(file_name, 'rb')
        if verbose:
            print("Parameters")

        # ------------------------------------------
        # Read the configuration file line by line
        while True:
            line2parse = config_file.readline()
            if verbose:
                print(line2parse, end='')

            # Quit parsing if we reach the end
            if not line2parse:
                break

            # Parse
            parse_res = parse(
                '{parse_type}: {field_name} = {read_value};{trash}',
                line2parse
            )

            # Skip if it is something we cannot parse
            if parse_res is not None:
                if parse_res['parse_type'] == 'ss':
                    setattr(self, parse_res['field_name'], parse_res[
                            'read_value'].split(','))
                elif parse_res['parse_type'] == 's':
                    setattr(self, parse_res['field_name'],
                            parse_res['read_value'])
                elif parse_res['parse_type'] == 'd':
                    setattr(self, parse_res['field_name'],
                            int(parse_res['read_value']))
                elif parse_res['parse_type'] == 'f':
                    setattr(self, parse_res['field_name'],
                            float(parse_res['read_value']))
                elif parse_res['parse_type'] == 'b':
                    setattr(self, parse_res['field_name'], bool(
                        int(parse_res['read_value'])))
                else:
                    print('  L-> skipped')
                    # raise RuntimeError('Unknown parse type!')


#
# custom_types.py ends here
