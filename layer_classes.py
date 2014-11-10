"""
Collection of layer types
"""
import numpy
from pylab import *
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

# This variant is the standard multiple input, multiple output version
class PoissonRegression(object):
    """
    Poisson Regression Class
    The poisson regression is fully described by a weight matrix :math: `W`
    and bias vector :math:`b`. 
    """

    def __init__(self, input, n_in, n_out):
        """
        Initialize the parameters of the poisson regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                 dtype=theano.config.floatX), name='W', borrow=True)

        # Initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=0*numpy.ones((n_out,),
                 dtype=theano.config.floatX), name='b', borrow=True)

        # Helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
                        dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
                        dtype=theano.config.floatX), name='b_helper', borrow=True)

        # Helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
                         dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
                         dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # Compute vector of expected values (for each output) in symbolic form
        self.E_y_given_x = T.exp(T.dot(input, self.W) + self.b)
        #self.E_y_given_x = T.exp(T.dot(input, self.W) + self.b) #possible alternative

        # Since predictions should technically be discrete, chose y which is most likely
        # (compute p(y) for multiple y values using E[y|x] computed above and select max)

        # self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Parameters of the model
        self.params         = [self.W, self.b]
        self.params_helper  = [self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2]

    def negative_log_likelihood(self, y, trialCount, maskData):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::
        p(y_observed|model,x_input) = E[y|x]^y exp(-E[y|x])/factorial(y)
        
        take sum over output neurons and times

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        return -T.sum(maskData * ((y * T.log(self.E_y_given_x)) - (trialCount * self.E_y_given_x)), axis=0)
        #return -T.sum(maskData *(T.log( (self.E_y_given_x.T ** y) * T.exp(-self.E_y_given_x.T) / T.gamma(y+1) )) )

    def errors(self, y, trialCount, maskData):
        """Use summed absolute value of difference between actual number of spikes per bin and predicted E[y|x]

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        return T.sum(maskData * T.sqrt(((trialCount*self.E_y_given_x)-y) ** 2))


# This variant takes in a number of trials argument as well.
class PoissonRegressionN(object):
    """Poisson Regression Class

    The poisson regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. 
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the poisson regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=0*numpy.ones((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # compute vector of expected values (for each output) in symbolic form
        self.E_y_given_x = T.exp(T.dot(input, self.W) + self.b)
        #self.E_y_given_x = T.exp(T.dot(input, self.W) + self.b) #possible alternative

        # since predictions should technically be discrete, chose y which is most likely (compute p(y) for multiple y values using E[y|x] computed above and select max)
        #self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]
        self.params_helper = [self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2]

    def negative_log_likelihood(self, y, trialCount, maskData):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::
        p(y_observed|model,x_input) = E[y|x]^y exp(-E[y|x])/factorial(y)
        
        take sum over output neurons and times

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        """
        return -T.sum( maskData * (  (y * T.log(self.E_y_given_x.T)) - (trialCount * self.E_y_given_x.T)  ) , axis = 0)
        #return -T.sum( maskData *(T.log( (self.E_y_given_x.T ** y) * T.exp(-self.E_y_given_x.T) / T.gamma(y+1) )) )

    def errors(self, y, trialCount, maskData):
        """Use summed absolute value of difference between actual number of spikes per bin and predicted E[y|x]

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        return T.sum(  maskData * T.sqrt(((trialCount*self.E_y_given_x.T)-y) ** 2)  )




# This class is to build the LeNet-style convolution + max pooling layers + output nonlinearity
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
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
        self.input = input

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
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)


        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros(filter_shape, \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((filter_shape[0],), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros(filter_shape, \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((filter_shape[0],), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # parameters of this layer
        self.params = [self.W, self.b]
        self.params_helper = [self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2]

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x');

        # Activation is given by sigmoid:
        #self.output = T.tanh(lin_output)

        # Activation is rectified linear
        self.output = lin_output*(lin_output>0)



# This class is to build an all-to-all hidden layer
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        an activation function (see below). Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

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
                           layer (overwritten in body of class)
        """
        self.input = input

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
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # helper variables for adagrad
        self.W_helper = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper', borrow=True)
        self.b_helper = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper', borrow=True)

        # helper variables for L1
        self.W_helper2 = theano.shared(value=numpy.zeros((n_in, n_out), \
            dtype=theano.config.floatX), name='W_helper2', borrow=True)
        self.b_helper2 = theano.shared(value=numpy.zeros((n_out,), \
            dtype=theano.config.floatX), name='b_helper2', borrow=True)

        # parameters of this layer
        self.params = [self.W, self.b]
        self.params_helper = [self.W_helper, self.b_helper]
        self.params_helper2 = [self.W_helper2, self.b_helper2]

        lin_output = T.dot(input, self.W) + self.b

        # Hidden unit activation is given by: tanh(dot(input,W) + b)
        #self.output = T.tanh(lin_output)

        # Hidden unit activation is rectified linear
        self.output = lin_output*(lin_output>0)

        # Hidden unit activation is None (i.e. linear)
        #self.output = lin_output



# This class is to build a E-I hidden layer
# W_E is weights from input to excitatory, W_I is weights from input to inhibitory
# W_EI is weights from E to I, W_IE is other direction
class HiddenLayerEI(object):
    def __init__(self, rng, input, n_in, n_outE, n_outI):
        """
        RNN hidden layer: units are fully-connected and have
        an activation function (see below). Wights project inputs to the units which are recurrently connected. 
        Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_outE: int
        :param n_outE: number of hidden excitatory units

        :type n_outI: int
        :param n_outI: number of hidden inhibitory units
        """
        self.input = input

        # Make all weights positive and when updating them, project to zero (so they remain positive)
        W_E_values = numpy.asarray(rng.uniform(low=0, high=.01, size=(n_in, n_outE)), dtype=theano.config.floatX)
        W_E = theano.shared(value=W_E_values, name='W_E', borrow=True)
        W_I_values = numpy.asarray(rng.uniform(low=0, high=.01, size=(n_in, n_outI)), dtype=theano.config.floatX)
        W_I = theano.shared(value=W_I_values, name='W_I', borrow=True)

        b_E_values = numpy.zeros((n_outE,), dtype=theano.config.floatX)
        b_E = theano.shared(value=b_E_values, name='b_E', borrow=True)
        b_I_values = numpy.zeros((n_outI,), dtype=theano.config.floatX)
        b_I = theano.shared(value=b_I_values, name='b_I', borrow=True)

        W_EI_values = numpy.asarray(rng.uniform(low=0, high=0.0001, size=(n_outE, n_outI)), dtype=theano.config.floatX)
        W_EI = theano.shared(value=W_EI_values, name='W_EI', borrow=True)
        W_IE_values = numpy.asarray(rng.uniform(low=0, high=0.0001, size=(n_outI, n_outE)), dtype=theano.config.floatX)
        W_IE = theano.shared(value=W_IE_values, name='W_IE', borrow=True)

        self.W_E = W_E
        self.W_I = W_I
        self.b_E = b_E
        self.b_I = b_I
        self.W_EI = W_EI
        self.W_IE = W_IE

        # helper variables for adagrad
        self.W_E_helper = theano.shared(value=numpy.zeros((n_in, n_outE), \
            dtype=theano.config.floatX), name='W_E_helper', borrow=True)
        self.W_I_helper = theano.shared(value=numpy.zeros((n_in, n_outI), \
            dtype=theano.config.floatX), name='W_I_helper', borrow=True)
        self.W_EI_helper = theano.shared(value=numpy.zeros((n_outE, n_outI), \
            dtype=theano.config.floatX), name='W_EI_helper', borrow=True)
        self.W_IE_helper = theano.shared(value=numpy.zeros((n_outI, n_outE), \
            dtype=theano.config.floatX), name='W_IE_helper', borrow=True)
        self.b_E_helper = theano.shared(value=numpy.zeros((n_outE,), \
            dtype=theano.config.floatX), name='b_E_helper', borrow=True)
        self.b_I_helper = theano.shared(value=numpy.zeros((n_outI,), \
            dtype=theano.config.floatX), name='b_I_helper', borrow=True)

        # helper variables for L1
        self.W_E_helper2 = theano.shared(value=numpy.zeros((n_in, n_outE), \
            dtype=theano.config.floatX), name='W_E_helper2', borrow=True)
        self.W_I_helper2 = theano.shared(value=numpy.zeros((n_in, n_outI), \
            dtype=theano.config.floatX), name='W_I_helper2', borrow=True)
        self.W_EI_helper2 = theano.shared(value=numpy.zeros((n_outE, n_outI), \
            dtype=theano.config.floatX), name='W_EI_helper2', borrow=True)
        self.W_IE_helper2 = theano.shared(value=numpy.zeros((n_outI, n_outE), \
            dtype=theano.config.floatX), name='W_IE_helper2', borrow=True)
        self.b_E_helper2 = theano.shared(value=numpy.zeros((n_outE,), \
            dtype=theano.config.floatX), name='b_E_helper2', borrow=True)
        self.b_I_helper2 = theano.shared(value=numpy.zeros((n_outI,), \
            dtype=theano.config.floatX), name='b_I_helper2', borrow=True)

        # Parameters of this layer
        self.params         = [self.W_EI, self.W_IE, self.W_E, self.W_I, self.b_E, self.b_I]
        self.params_helper  = [self.W_EI_helper, self.W_IE_helper, self.W_E_helper, self.W_I_helper, self.b_E_helper, self.b_I_helper]
        self.params_helper2 = [self.W_EI_helper2, self.W_IE_helper2, self.W_E_helper2, self.W_I_helper2, self.b_E_helper2, self.b_I_helper2]

        # Initial hidden state values
        h_E_0 = T.zeros((n_outE,)) 
        h_I_0 = T.zeros((n_outI,))

        # Recurrent function with rectified linear output activation function (u is input, h is hidden activity)
        def step(u_t, h_E_tm1, h_I_tm1):
            lin_E = T.dot(u_t, self.W_E) - T.dot(h_I_tm1, self.W_IE) + self.b_E
            h_E_t = lin_E*(lin_E>0)
            lin_I = T.dot(u_t, self.W_I) + T.dot(h_E_tm1, self.W_EI) + self.b_I
            h_I_t = lin_I*(lin_I>0)
            return h_E_t, h_I_t

        # Compute the hidden E & I timeseries
        [h_E, h_I], _ = theano.scan(step,
                        sequences=self.input,
                        outputs_info=[h_E_0, h_I_0])

        # Output activity is the hidden unit activity
        self.h_E = h_E
        self.h_I = h_I



