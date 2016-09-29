# models.py ---
#
# Filename: models.py
# Description: Python Module With the models to be used
# Author: Kwang
# Maintainer:
# Created: Tue Jan 27 11:17:55 2015 (+0100)
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

import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import networks.custom_theano as CT

floatX = theano.config.floatX
NORMAL_INIT_SIGMA = 0.5

ATANPOOL_IN = 1
ATANPOOL_OUT = 2
ATANPOOL_INOUT = 3


class LeNetConvPoolLayer(object):
    """ LeNet Conv Pool Layer modified from Theano Example """

    def __init__(self, rng, input, filter_shape, image_shape,
                 poolsize=(2, 2), activation=T.tanh, W_in=None, b_in=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape

        self.name = "LeNetConvPoolLayer"
        self.args_in = (rng, input, filter_shape, image_shape,
                        poolsize, activation, W_in, b_in)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))

        if W_in is None:
            if False:
                # original initialization
                self.W = theano.shared(numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound,
                                size=self.filter_shape),
                    dtype=floatX), name='W', borrow=True)
            # Gaussian Initialization
            self.W = theano.shared(numpy.asarray(
                rng.normal(scale=NORMAL_INIT_SIGMA,
                           size=self.filter_shape),
                dtype=floatX), name='W', borrow=True)
        else:
            self.W = W_in

        # the bias is a 1D tensor -- one bias per output feature map
        if b_in is None:
            b_values = numpy.zeros((filter_shape[0],), dtype=floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            self.b = b_in

        # convolve input feature maps with filters
        self.conv_out = conv.conv2d(input=input, filters=self.W,
                                    filter_shape=filter_shape,
                                    image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=self.conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = (
            lin_output if activation is None else activation(lin_output))

        # store parameters of this layer
        self.params = [self.W, self.b]

    def saveLayer(self, file_name):
        numpy.save(file_name + '_W.npy', self.W.get_value())
        numpy.save(file_name + '_b.npy', self.b.get_value())

    def loadLayer(self, file_name):
        self.W.set_value(numpy.load(file_name + '_W.npy'))
        self.b.set_value(numpy.load(file_name + '_b.npy'))


class HiddenLayer(object):
    """ Hidden Layer class modified from Theano Example.
    """

    def __init__(self, rng, input, n_in, n_out,
                 activation=T.tanh, nAtanPool=0, W_in=None, b_in=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.name = 'HiddenLayer'
        self.args_in = (rng, input, n_in, n_out,
                        activation, nAtanPool, W_in, b_in)

        # ---------------------------------------------------------------------
        # Resetting the n_in and n_out
        if nAtanPool == ATANPOOL_IN or nAtanPool == ATANPOOL_INOUT:
            assert n_in % 2 == 0
            n_in = n_in / 2
        if nAtanPool == ATANPOOL_OUT or nAtanPool == ATANPOOL_INOUT:
            n_out = n_out * 2

        # If nAtanPool is -1 then it means in this layer, we atan pool
        # on the input (which will reduce the number of input nodes to
        # half...
        if nAtanPool == ATANPOOL_IN or nAtanPool == ATANPOOL_INOUT:
            input = CT.custom_arctan2(input[:, :n_in], input[:, n_in:])

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W_in is None:
            if False:
                # original initialization
                W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
                if activation == theano.tensor.nnet.sigmoid:
                    W_values *= 4
                self.W = theano.shared(value=W_values, name='W', borrow=True)
            # Gaussian Initialization
            self.W = theano.shared(numpy.asarray(
                rng.normal(scale=NORMAL_INIT_SIGMA,
                           size=(n_in, n_out)),
                dtype=floatX), name='W', borrow=True)

        else:
            self.W = W_in
            # W_values = numpy.asarray(Wi, dtype=theano.config.floatX)
            # if activation == theano.tensor.nnet.sigmoid:
            #     W_values *= 4

        if b_in is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            # b_values = numpy.asarray(bi, dtype=theano.config.floatX)
            self.b = b_in

        lin_output = T.dot(input, self.W) + self.b
        output = (lin_output if activation is None else activation(lin_output))

        # If nAtanPool is 2 then it means in this layer, we atan pool
        # on the output (which will reduce the number of output nodes to
        # half...
        if nAtanPool == ATANPOOL_OUT or nAtanPool == ATANPOOL_INOUT:
            output = CT.custom_arctan2(output[:, :n_out], output[:, n_out:])

        self.output = output

        # parameters of the model
        self.params = [self.W, self.b]

    def saveLayer(self, file_name):
        numpy.save(file_name + '_W.npy', self.W.get_value())
        numpy.save(file_name + '_b.npy', self.b.get_value())

    def loadLayer(self, file_name):
        self.W.set_value(numpy.load(file_name + '_W.npy'))
        self.b.set_value(numpy.load(file_name + '_b.npy'))


class PReLUHiddenLayer(object):
    """ Hidden Layer implementation adapted for PReLU """

    def __init__(self, rng, input, n_in, n_out,
                 nAtanPool=0, W_in=None, alpha_in=None, b_in=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.name = 'PReLUHiddenLayer'
        self.args_in = (rng, input, n_in, n_out,
                        nAtanPool, W_in, alpha_in, b_in)

        # ---------------------------------------------------------------------
        # Resetting the n_in and n_out
        if nAtanPool == ATANPOOL_IN or nAtanPool == ATANPOOL_INOUT:
            assert n_in % 2 == 0
            n_in = n_in / 2
        if nAtanPool == ATANPOOL_OUT or nAtanPool == ATANPOOL_INOUT:
            n_out = n_out * 2

        # If nAtanPool is -1 then it means in this layer, we atan pool
        # on the input (which will reduce the number of input nodes to
        # half...
        if nAtanPool == ATANPOOL_IN or nAtanPool == ATANPOOL_INOUT:
            input = CT.custom_arctan2(input[:, :n_in], input[:, n_in:])

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W_in is None:
            # Gaussian Initialization
            self.W = theano.shared(numpy.asarray(
                rng.normal(scale=NORMAL_INIT_SIGMA,
                           size=(n_in, n_out)),
                dtype=floatX), name='W', borrow=True)

        else:
            self.W = W_in

        if alpha_in is None:
            self.alpha = theano.shared(numpy.asarray(rng.normal(
                scale=NORMAL_INIT_SIGMA, size=(n_out,)
            ), dtype=floatX), name='alpha', borrow=True)
        else:
            self.alpha = alpha_in

        if b_in is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            self.b = b_in

        lin_output = T.dot(input, self.W) + self.b

        pos_output = lin_output * (lin_output > 0)  # Relu
        neg_output = self.alpha * lin_output * \
            (lin_output <= 0)  # Relu the other way

        output = pos_output + neg_output

        # If nAtanPool is 2 then it means in this layer, we atan pool
        # on the output (which will reduce the number of output nodes to
        # half...
        if nAtanPool == ATANPOOL_OUT or nAtanPool == ATANPOOL_INOUT:
            output = CT.custom_arctan2(output[:, :n_out], output[:, n_out:])

        self.output = output

        # parameters of the model
        self.params = [self.W, self.alpha, self.b]

    def saveLayer(self, file_name):
        numpy.save(file_name + '_W.npy', self.W.get_value())
        numpy.save(file_name + '_alpha.npy', self.alpha.get_value())
        numpy.save(file_name + '_b.npy', self.b.get_value())

    def loadLayer(self, file_name):
        self.W.set_value(numpy.load(file_name + '_W.npy'))
        self.alpha.set_value(numpy.load(file_name + '_alpha.npy'))
        self.b.set_value(numpy.load(file_name + '_b.npy'))


class GHHHiddenLayer(object):

    """ Hidden Layer using GHH activation. Modified from Theano Example. """

    def __init__(self, rng, input, n_in, n_out, n_sum, n_max,
                 batch_size,
                 delta_i=None,
                 activation=T.tanh, nAtanPool=0, W_in=None, b_in=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.name = 'GHHHiddenLayer'
        self.args_in = (rng, input, n_in, n_out, n_sum, n_max,
                        batch_size, delta_i, activation, nAtanPool, W_in, b_in)

        # ---------------------------------------------------------------------
        # Resetting the n_in and n_out
        if nAtanPool == ATANPOOL_IN or nAtanPool == ATANPOOL_INOUT:
            assert n_in % 2 == 0
            n_in = n_in / 2
        if nAtanPool == ATANPOOL_OUT or nAtanPool == ATANPOOL_INOUT:
            n_out = n_out * 2

        # If nAtanPool is -1 then it means in this layer, we atan pool
        # on the input (which will reduce the number of input nodes to
        # half...
        if nAtanPool == ATANPOOL_IN or nAtanPool == ATANPOOL_INOUT:
            input = CT.custom_arctan2(input[:, :n_in], input[:, n_in:])

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W_in is None:
            if False:
                # original initialization (not used)
                W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out * n_sum * n_max)),
                    high=numpy.sqrt(6. / (n_in + n_out * n_sum * n_max)),
                    size=(n_in, n_out, n_sum, n_max)
                ), dtype=theano.config.floatX)
                if activation == theano.tensor.nnet.sigmoid:
                    W_values *= 4
                self.W = theano.shared(value=W_values, name='W', borrow=True)

            # Gaussian Initialization
            self.W = theano.shared(numpy.asarray(
                rng.normal(scale=NORMAL_INIT_SIGMA,
                           size=(n_in, n_out, n_sum, n_max)),
                dtype=floatX), name='W', borrow=True)

        else:
            self.W = W_in
            # W_values = numpy.asarray(Wi, dtype=theano.config.floatX)
            # if activation == theano.tensor.nnet.sigmoid:
            #     W_values *= 4

        # the bias
        if b_in is None:
            b_values = numpy.zeros((n_out, n_sum, n_max,), dtype=floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)
        else:
            self.b = b_in

        if delta_i is None:
            # the deltas...
            if n_sum > 1:
                assert n_sum % 2 == 0
                delta_values = numpy.tile(numpy.asarray(
                    [1, -1], dtype=floatX), reps=[n_sum / 2])
            else:
                delta_values = numpy.asarray([1])
            self.delta = theano.shared(delta_values, name='delta', borrow=True)
        else:
            assert delta_i.shape[0] == n_sum
            self.delta = theano.shared(delta_i, name='delta', borrow=True)

        W_vec = self.W.reshape([n_in, n_out * n_sum * n_max])
        raw_out = T.dot(input, W_vec) + self.b.flatten().dimshuffle('x', 0)
        self.reshaped_raw_out = raw_out.reshape([batch_size, n_out, n_sum, n_max])

        self.max_out = T.max(self.reshaped_raw_out, axis=3)
        self.sum_out = T.sum(
            self.max_out * self.delta.dimshuffle('x', 'x', 0),
            axis=2, dtype=floatX
        )

        lin_output = self.sum_out
        output = (lin_output if activation is None else activation(lin_output))

        # If nAtanPool is 2 then it means in this layer, we atan pool
        # on the output (which will reduce the number of output nodes to
        # half...
        if nAtanPool == ATANPOOL_OUT or nAtanPool == ATANPOOL_INOUT:
            output = CT.custom_arctan2(output[:, :n_out], output[:, n_out:])

        self.output = output

        # parameters of the model
        self.params = [self.W, self.b]

    def saveLayer(self, file_name):
        numpy.save(file_name + '_W.npy', self.W.get_value())
        numpy.save(file_name + '_b.npy', self.b.get_value())

    def loadLayer(self, file_name):
        self.W.set_value(numpy.load(file_name + '_W.npy'))
        self.b.set_value(numpy.load(file_name + '_b.npy'))


def _dropout_from_layer(rng, layer, p):
    """ For Dropout

    parts of the implementation are taken from
    https://github.com/mdenil/dropout/blob/master/mlp.py

    Notes: p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


class DropoutHiddenLayer(HiddenLayer):
    """ Dropout Version of the Hidden layer """

    def __init__(self, dropout_rate, rng, input, n_in, n_out,
                 activation=T.tanh, nAtanPool=0, W_in=None, b_in=None):
        super(DropoutHiddenLayer, self).__init__(
            rng=rng, input=input, n_in=n_in, n_out=n_out,
            activation=activation, nAtanPool=nAtanPool,
            W_in=W_in, b_in=b_in)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class DropoutPReLUHiddenLayer(PReLUHiddenLayer):
    """ Dropout Version of the PReLUHidden layer """

    def __init__(self, dropout_rate, rng, input, n_in, n_out,
                 nAtanPool=0, W_in=None, alpha_in=None, b_in=None):
        super(DropoutPReLUHiddenLayer, self).__init__(
            rng=rng, input=input, n_in=n_in, n_out=n_out, nAtanPool=nAtanPool,
            W_in=W_in, alpha_in=alpha_in, b_in=b_in)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class DropoutLeNetConvPoolLayer(LeNetConvPoolLayer):
    """ Dropout Version of the LeNetConvPoolLayer layer """

    def __init__(self, dropout_rate, rng, input, filter_shape, image_shape,
                 poolsize=(2, 2), activation=T.tanh, W_in=None, b_in=None):
        super(DropoutLeNetConvPoolLayer, self).__init__(
            rng=rng, input=input, filter_shape=filter_shape,
            image_shape=image_shape, poolsize=poolsize,
            activation=activation, W_in=W_in, b_in=b_in)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class DropoutGHHHiddenLayer(GHHHiddenLayer):
    """ Dropout Version of the GHH Hidden layer """

    def __init__(self, dropout_rate, rng, input, n_in, n_out, n_sum, n_max,
                 batch_size,
                 delta_i=None,
                 activation=T.tanh, nAtanPool=0, W_in=None, b_in=None):
        super(DropoutGHHHiddenLayer, self).__init__(
            rng=rng, input=input, n_in=n_in, n_out=n_out,
            n_sum=n_sum, n_max=n_max,
            batch_size=batch_size, delta_i=delta_i,
            activation=activation, nAtanPool=nAtanPool,
            W_in=W_in, b_in=b_in)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


#
# models.py ends here
