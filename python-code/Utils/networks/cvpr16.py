# cvpr16.py ---
#
# Filename: cvpr16.py
# Description:
# Author: Kwang Moo Yi
# Maintainer: Kwang Moo Yi
# Created: Fri Jul  1 11:15:22 2016 (-0700)
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

import os
import sys
# Disable future warnings (caused by theano)
import warnings

import numpy
import theano
import theano.tensor as T

warnings.simplefilter(action="ignore", category=FutureWarning)

floatX = theano.config.floatX
bUseWNN = True


class SiameseOrientationLearnerConfig:

    def __init__(self):

        # Placeholders
        self.images = ''
        self.save_dir = ''
        self.batch_size = 0
        self.patch_size = 0
        self.num_channel = 0

        # 0 for all shuffle, 1 for batchwise
        self.shuffle_type = 0
        # number of siamese branches
        self.num_siamese = 1
        self.modelType = 'CNNGHHH'
        self.mapping = 'Arctan'

        self.num_proc = 4

        self.GHH_numSUM = None
        self.GHH_numMAX = None


class SiameseOrientationLearner(object):
    '''
    classdocs
    '''

    def __init__(self, config, rng=None):
        '''
        Constructor

        '''
        if rng is None:
            self.rng = numpy.random.RandomState(23455)
        else:
            self.rng = rng

        # ---------------------------------------------------------------------
        # read config
        self.config = config

        # ---------------------------------------------------------------------
        # Pre-allocate empty lists for multiple instances of the same layers
        self.layers = [
            None for _ in xrange(self.config.num_siamese)
        ]
        self.orientation_output = [
            None for _ in xrange(self.config.num_siamese)
        ]
        self.test_layers = [
            None for _ in xrange(self.config.num_siamese)
        ]
        self.test_orientation_output = [
            None for _ in xrange(self.config.num_siamese)
        ]

        # ---------------------------------------------------------------------
        # Theano Variables

        # N x D x P x P : data (P is the patch size, D is the number of
        # channels)
        self.x = [None for _ in xrange(self.config.num_siamese)]

        for idxSiam in xrange(self.config.num_siamese):
            # N x D x P x P : data (P is the patch size, D is the number of
            # channels)
            self.x[idxSiam] = T.tensor4('x_' + str(idxSiam), dtype=floatX)

        theano.config.exception_verbosity = 'high'

    def setupDataAndCompile4Test(self):

        # ---------------------------------------------------------------------
        # Allocate Theano Shared Variables (allocated on the GPU)
        batch_size = self.config.batch_size
        patch_size = self.config.patch_size
        num_channel = self.config.num_channel

        self.test_x = [None for _ in xrange(self.config.num_siamese)]

        for idxSiam in xrange(self.config.num_siamese):
            self.test_x[idxSiam] = theano.shared(
                numpy.zeros(
                    (batch_size, num_channel, patch_size, patch_size),
                    dtype=floatX
                ), name='test_x_' + str(idxSiam), borrow=True)

        # ---------------------------------------------------------------------
        # Compile Functions for Training

        # setup givens (TODO: consider using the macro batch stuff)
        givens_test = {}
        for idxSiam in xrange(self.config.num_siamese):
            givens_test[self.x[idxSiam]] = self.test_x[idxSiam]

        print("compiling get_output() ... ", end="")
        sys.stdout.flush()
        self.get_output = theano.function(
            inputs=[],
            outputs=self.test_orientation_output[0],
            givens={self.x[0]: self.test_x[0]}
        )
        print("done.")

        print("compiling get_raw_output() ... ", end="")
        sys.stdout.flush()
        self.get_raw_output = theano.function(
            inputs=[],
            outputs=self.test_layers[0][-1].output,
            givens={self.x[0]: self.test_x[0]}
        )
        print("done.")

    def runTest(self, x_in, model_epoch=""):
        '''
        The test loop

        '''

        # Check if setup is done

        # The main training loop (copy data manually using set_value to the
        # shared variables)

        # Read parameters
        batch_size = self.config.batch_size
        num_siamese = 1         # just use the first siamese only

        save_dir = self.config.save_dir

        # Check if all data fits into a single batch
        assert batch_size == len(x_in)

        test_x_in = x_in        # should not be in the siamese form
        # test_x_id_in = x_id_in
        # test_y_in = y_in

        # ---------------------------------------------------------------------
        # Testing Loop
        print('testing...', end="")
        sys.stdout.flush()
        # ---------------------------------------------------------------------
        # If the dump directory exists
        if os.path.exists(save_dir + model_epoch):

            # for each layer of the model
            for idxLayer in xrange(len(self.layers[0])):
                save_file_name = save_dir + \
                    model_epoch + "layer" + str(idxLayer)

                for idxSiam in xrange(num_siamese):
                    self.layers[idxSiam][idxLayer].loadLayer(save_file_name)

        else:
            print(("PATH:", save_dir + model_epoch))
            raise NotImplementedError("I DON'T HAVE THE LEARNED MODEL READY!")

        # Copy test data to share memory (test memory)
        for idxSiam in xrange(num_siamese):
            self.test_x[idxSiam].set_value(
                numpy.asarray(test_x_in, dtype=floatX))

        test_result = self.get_output()
        test_raw_output = self.get_raw_output()
        # ---------------------------------------------------------------------
        print(" done!")

        return test_result, test_raw_output

#
# cvpr16.py ends here
