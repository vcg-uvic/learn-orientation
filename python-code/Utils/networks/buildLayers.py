# buildLayers.py ---
#
# Filename: buildLayers.py
# Description:
# Author: Kwang
# Maintainer:
# Created: Fri Aug 14 13:44:40 2015 (+0200)
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

from __future__ import print_function

# Disable future warnings (caused by theano)
import warnings

import theano

# for the custom theano code
import custom_theano as CT
from Utils.models import (DropoutGHHHiddenLayer, DropoutHiddenLayer,
                          DropoutLeNetConvPoolLayer, GHHHiddenLayer,
                          HiddenLayer, LeNetConvPoolLayer)

warnings.simplefilter(action="ignore", category=FutureWarning)

floatX = theano.config.floatX


def instantiateLayers(myNet, deterministic):
    '''
    Build layers here

    '''

    # ------------------------------------------------------------------------
    # Run appropriate build layer list function
    bDropout = myNet.config.modelType[:7] == 'Dropout'
    if bDropout:
        buildName = myNet.config.modelType[7:]
    else:
        buildName = myNet.config.modelType
    build_function_name = "buildNameArg" + buildName
    layer_name_list, layer_arg_list, dropout_rates = globals(
    )[build_function_name](myNet, bDropout)

    # ------------------------------------------------------------------------
    # Layer construction for Siamese
    for idxSiam in xrange(myNet.config.num_siamese):

        # Instantiation of the Network Layers
        if bDropout:
            myNet.layers[idxSiam] = buildLayersList(
                layer_name_list, layer_arg_list, myNet.x[idxSiam],
                myNet.rng, dropout_rates
            )
        else:
            myNet.layers[idxSiam] = buildLayersList(
                layer_name_list, layer_arg_list, myNet.x[idxSiam], myNet.rng
            )

        # ---------------------------------------------------------------------
        # Instantiation of the Output Layer
        if myNet.config.mapping.lower() != "arctan":
            raise NotImplementedError(
                "Only supports Arctan mapping."
            )

        # Solution with custom arctan2
        myNet.orientation_output[idxSiam] = CT.custom_arctan2(
            myNet.layers[idxSiam][-1].output.flatten(2)[:, 0],
            myNet.layers[idxSiam][-1].output.flatten(2)[:, 1]
        )

        # ---------------------------------------------------------------------
        # If necessary, generate the dropout test layer and the corresponding
        # output
        if bDropout and deterministic:
            myNet.test_layers[idxSiam] = buildDropoutTestLayers(
                myNet.layers[idxSiam], dropout_rates)
        else:
            myNet.test_layers[idxSiam] = myNet.layers[idxSiam]

        # ---------------------------------------------------------------------
        # Instantiation of the Output Layer for testing
        # Solution with custom arctan2
        myNet.test_orientation_output[idxSiam] = CT.custom_arctan2(
            myNet.test_layers[idxSiam][-1].output.flatten(2)[:, 0],
            myNet.test_layers[idxSiam][-1].output.flatten(2)[:, 1]
        )

    # Initialize with same value for each copy
    for idxSiam in xrange(1, myNet.config.num_siamese):
        for cur_layer, ref_layer in zip(
                myNet.layers[idxSiam], myNet.layers[0]
        ):
            for param, ref_param in zip(cur_layer.params, ref_layer.params):
                param.set_value(ref_param.get_value())


def buildNameArgCNNGHHH(myNet, bDropout=False):
    '''
    Build layers here

    '''
    batch_size = myNet.config.batch_size

    # ------------------------------------------------------------------------
    # Dropout Ratio Settings
    # Conv/Conv/Conv/FC/FC
    dropout_rates = [0.0, 0.0, 0.0, 0.3, 0.0]

    # ------------------------------------------------------------------------
    # Activation settings

    relu = lambda x: x * (x > 0)  # from theano example

    conv_activation = relu
    fc_activation = None
    out_activation = None

    output_type = myNet.config.mapping
    if output_type == 'Arctan':
        output_dim = 2
    else:
        output_dim = 1

    # ------------------------------------------------------------------------
    # CNN Settings

    # # ----------- Conv-Pool Layers ----------
    # # 64x64 --> 60x60 --> 20x20
    # # 20x20 --> 16x16 --> 4x4
    # # 4x4 --> 2x2 --> 1x1
    # kernSZ_list = [5,5,3]
    # poolSZ_list = [(3,3),(4,4),(2,2)]
    # nkerns_list = [10,20,50]

    # # -------- Fully Connected Layers --------
    # nNumHidden_list = [nkerns_list[-1], 500 ,output_dim ] # should always
    # start with nkerns_list[-1] and end with output_dim

    # ----------- Conv-Pool Layers ----------
    # 28x28 --> 24x24 --> 12x12
    # 12x12 --> 8x8 --> 4x4
    # 4x4 --> 2x2 --> 1x1
    kernSZ_list = [5, 5, 3]
    poolSZ_list = [(2, 2), (2, 2), (2, 2)]
    nkerns_list = [10, 20, 50]

    # -------- Fully Connected Layers --------
    nNumHiddenNode = 20
    if 'nNumHiddenNode' in myNet.config.__dict__.keys():
        nNumHiddenNode = myNet.config.nNumHiddenNode
    # should always start with nkerns_list[-1] and end with output_dim
    nNumHidden_list = [nkerns_list[-1], nNumHiddenNode, output_dim]
    numSum = myNet.config.GHH_numSUM
    numMax = myNet.config.GHH_numMAX

    patchSZ_list = [None] * len(kernSZ_list)
    patchSZ_list[0] = myNet.config.patch_size
    for i in xrange(1, len(patchSZ_list)):
        patchSZ_list[i] = (patchSZ_list[i - 1] -
                           (kernSZ_list[i - 1] - 1)) / poolSZ_list[i - 1][0]

    img_dim_list = [myNet.config.num_channel] + nkerns_list

    # ------------------------------------------------------------------------
    # Build the list of layer names and arguments
    layer_name_list = []
    layer_arg_list = []

    # Settings for the Conv-Pool Layers
    for idxL in xrange(len(kernSZ_list)):
        # name
        cur_layer_name = "LeNetConvPoolLayer"
        if bDropout:
            cur_layer_name = "Dropout" + cur_layer_name
        # arguments
        filter_shape = [nkerns_list[idxL], img_dim_list[
            idxL]] + [kernSZ_list[idxL]] * 2
        image_shape = [batch_size, img_dim_list[
            idxL]] + [patchSZ_list[idxL]] * 2
        cur_layer_arg = (filter_shape, image_shape, poolSZ_list[
                         idxL], conv_activation, None, None)
        # store to list
        layer_name_list += [cur_layer_name]
        layer_arg_list += [cur_layer_arg]

    # Settings for the Fully Connected Layers (Using GHH!)
    for idxL in xrange(len(nNumHidden_list) - 1):
        # name
        cur_layer_name = "GHHHiddenLayer"
        if bDropout:
            cur_layer_name = "Dropout" + cur_layer_name
        # argument
        if idxL == len(nNumHidden_list) - 2:
            cur_layer_arg = (
                nNumHidden_list[idxL],
                nNumHidden_list[idxL + 1],
                numSum, numMax, batch_size,
                None, out_activation, 0, None, None
            )
        else:
            cur_layer_arg = (
                nNumHidden_list[idxL],
                nNumHidden_list[idxL + 1],
                numSum, numMax, batch_size,
                None, fc_activation, 0, None, None
            )

        # store to list
        layer_name_list += [cur_layer_name]
        layer_arg_list += [cur_layer_arg]

    return layer_name_list, layer_arg_list, dropout_rates


# ------------------------------------------------------------------------
# Function for Creating Layers given names and argument lists
def buildLayersList(layer_name_list, layer_arg_list,
                    first_layer_input, rng, dropout_rates=None):
    # NOTE: dropout_rates will be used only if we are using dropout layers

    layers_list = []
    prev_layer_output = first_layer_input
    for idxLayer in xrange(len(layer_name_list)):

        cur_layer_name = layer_name_list[idxLayer]

        # flatten the input layer into two dims if hidden layer
        if cur_layer_name[-11:] == "HiddenLayer":
            cur_layer_input = prev_layer_output.flatten(2)
        else:
            cur_layer_input = prev_layer_output

        # append the random seed and the input to the arguments
        cur_layer_arg = (rng, cur_layer_input) + layer_arg_list[idxLayer]

        # Add dropout rates if we are using them
        if dropout_rates is not None:
            cur_layer_arg = (dropout_rates[idxLayer],) + cur_layer_arg

        cur_layer = globals()[cur_layer_name](*cur_layer_arg)
        layers_list += [cur_layer]
        prev_layer_output = cur_layer.output

    return layers_list


# ------------------------------------------------------------------------
# Function for Creating Duplicate Computation Graph for Testing Dropout
def buildDropoutTestLayers(orig_layer_list, dropout_rates):

    test_layer_list = []

    # For each layer in the network
    prev_layer_output = None
    for cur_layer, dropout_rate in zip(orig_layer_list, dropout_rates):

        # Note that these will carry the normal name and args_in excluding the
        # dropout init
        layer_name = cur_layer.name
        layer_args = cur_layer.args_in

        # Use input from the previous test layer
        if prev_layer_output is not None:
            # flatten the input layer into two dims if hidden layer
            if cur_layer.name[-11:] == "HiddenLayer":
                cur_layer_input = prev_layer_output.flatten(2)
            else:
                cur_layer_input = prev_layer_output
            layer_args = (layer_args[0],) + \
                (cur_layer_input,) + layer_args[2:]

        # Change the last arguments so that it simply uses the weights
        # of the  other layer!  note that the  original implementation
        # did not multiply the output for some reason
        new_layer_args = layer_args[:-2] + (
            cur_layer.W * (1 - dropout_rate), cur_layer.b * (1 - dropout_rate)
        )

        # Call the appropriate layer with it's name
        test_layer = globals()[layer_name](*new_layer_args)

        # Store the output
        prev_layer_output = test_layer.output

        test_layer_list += [test_layer]

    return test_layer_list

#
# buildLayers.py ends here
