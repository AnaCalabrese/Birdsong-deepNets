"""
Single-layer neural network with poisson output units
Code based on MLP tutorial code for theano
"""
__docformat__ = 'restructedtext en'

# import cPickle
import pickle
import gzip
import os
import sys
import time

import numpy
import h5py
from pylab import *
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

# Load data
from loading_functions import load_anas_data

#
from layer_classes import PoissonRegression, HiddenLayerEI, HiddenLayer

def SGD_training(learning_rate=1e-3, L1_reg=0, L2_reg=0, n_epochs=1000):
    """
    stochastic gradient descent optimization for a multilayer
    perceptron

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    dataset_info = load_anas_data()

    data_set_x, data_set_y = dataset_info[0]
    y_counts               = dataset_info[1]
    y_mask                 = dataset_info[2]
    maxBatchSize           = numpy.int_(dataset_info[3])

    batch_size      = maxBatchSize
    n_train_batches = 28

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x     = T.matrix('x')  # the data is presented as a vector of inputs with many exchangeable examples of this vector
    y     = T.matrix('y')  # the output is a vector of matched output unit responses.
    tc    = T.matrix('tc') # counts for how many trials each element of y is summed over
    md    = T.matrix('md') # a binary mask for whether or not to include this y datum in the NLL computation

    rng   = numpy.random.RandomState(1234)

    ################################################
    # Architecture 1: input --> hidden --> Poisson
    ################################################
    # Construct a fully-connected rectified linear layer
    # layerh = HiddenLayer(rng, input=x, n_in=data_set_x.get_value(borrow=True).shape[1], n_out=50)

    # The poisson regression layer gets as input the hidden units of the hidden layer
    # Layer0 = PoissonRegression(input=layerh.output, n_in=50, n_out=data_set_y.get_value(borrow=True).shape[1])

    ################################################
    # Architecture 2: input --> hiddenEI --> Poisson
    ################################################

    # construct a fully-connected rectified linear layer
    layerh = HiddenLayerEI(rng, input=x, n_in=data_set_x.get_value(borrow=True).shape[1], n_outE=200, n_outI=200)

    # The poisson regression layer gets as input the hidden units
    # of the hidden layer
    Layer0 = PoissonRegression(input=layerh.h_E, n_in=200, n_out=data_set_y.get_value(borrow=True).shape[1]) 

    ################################################
    # Architecture 3: input --> Poisson
    ################################################

    # The poisson regression layer gets as input the hidden units of the hidden layer
    # Layer0 = PoissonRegression(
    #    input=x,
    #    n_in=data_set_x.get_value(borrow=True).shape[1],
    #    n_out=data_set_y.get_value(borrow=True).shape[1])

    # L1 norm ; one regularization option is to enforce L1 norm to
    # be small
    L1 = abs(Layer0.W).sum()

    # square of L2 norm ; one regularization option is to enforce
    # square of L2 norm to be small
    L2_sqr = (Layer0.W ** 2).sum()

    negative_log_likelihood = Layer0.negative_log_likelihood

    errors = Layer0.errors

    # create a list (concatenated) of all model parameters to be fit by gradient descent
    ################################################
    # Architecture 1,2 params
    ################################################
    #order: [self.W, self.b] + [self.W_EI, self.W_IE, self.W_E, self.W_I, self.b_E, self.b_I]
    params = Layer0.params + layerh.params
    params_helper = Layer0.params_helper + layerh.params_helper
    params_helper2 = Layer0.params_helper2 + layerh.params_helper2

    ################################################
    # Architecture 3 params
    ################################################
    #params = Layer0.params
    #params_helper = Layer0.params_helper
    #params_helper2 = Layer0.params_helper2

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = T.sum(negative_log_likelihood(y, tc, md)) \
         + L1_reg * L1 \
         + L2_reg * L2_sqr

    # Compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch. use cost or errors(y,tc,md) as output?
    test_model = theano.function(inputs=[index],
            outputs=[negative_log_likelihood(y,tc,md), Layer0.E_y_given_x, y],
            givens={
                x: data_set_x[index * batch_size:(index + 1) * batch_size],
                y: data_set_y[index * batch_size:(index + 1) * batch_size],
                tc: y_counts[index * batch_size:(index + 1) * batch_size],
                md: y_mask[index * batch_size:(index + 1) * batch_size]})

    # wanted to use below indexes and have different sized batches, but this didn't work
    #[int(batchBreaks[index]-1):int(batchBreaks[(index+1)]-1)]

    validate_model = theano.function(inputs=[index],
            outputs=numpy.sum(negative_log_likelihood(y,tc,md)),
            givens={
                x: data_set_x[index * batch_size:(index + 1) * batch_size],
                y: data_set_y[index * batch_size:(index + 1) * batch_size],
                tc: y_counts[index * batch_size:(index + 1) * batch_size],
                md: y_mask[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta (stored in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in params:
        #gparam = theano.map(lambda yi,tci,mdi: T.grad(cost(yi,tci,mdi), param), sequences=[y,tc,md])
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    #for param, gparam in zip(params, gparams):
    #    updates.append((param, param - learning_rate * gparam))
    # adagrad
    iter_count = theano.shared(value=numpy.ones((107,), dtype=theano.config.floatX), name='iter_count', borrow=True)
    L1_penalized = [0]
    larger_stepsize = [1]
    enforce_positive = [2, 3]
    param_index = 0
    rho = 1e-6
    for param, param_helper, param_helper2, gparam in zip(params, params_helper, params_helper2, gparams):
        updates.append((param_helper, param_helper + gparam ** 2)) # need sum of squares for learning rate
        updates.append((param_helper2, param_helper2 + gparam))    # need sum of gradients for L1 thresholding
        if param_index in L1_penalized:
            updates.append((param, T.maximum(0,T.sgn(T.abs_(param_helper2)/iter_count - L1_reg)) * (-T.sgn(param_helper2)*learning_rate*iter_count/(rho + (param_helper + gparam ** 2) ** 0.5) * (T.abs_(param_helper2)/iter_count - L1_reg))))
        elif param_index in larger_stepsize:
            updates.append((param, param - learning_rate*1e3 * gparam / (rho + (param_helper + gparam ** 2) ** 0.5)))
        elif param_index in enforce_positive:
            updates.append((param, T.maximum(0, param - learning_rate*1e3 * gparam / (rho + (param_helper + gparam ** 2) ** 0.5) )  ))
        else:
            updates.append((param, param - learning_rate * gparam / (rho + (param_helper + gparam ** 2) ** 0.5)))
        param_index += 1
    updates.append((iter_count, iter_count + 1))


    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: data_set_x[index * batch_size:(index + 1) * batch_size],
                y: data_set_y[index * batch_size:(index + 1) * batch_size],
                tc: y_counts[index * batch_size:(index + 1) * batch_size],
                md: y_mask[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    #patience = train_set_x.get_value(borrow=True).shape[0] * n_epochs #no early stopping
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.999  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter  = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            print(minibatch_avg_cost)
 
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute absolute error loss on validation set
                validation_losses = [validate_model(i) for i
                                     in [28]]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i, validation error %f' %
                     (epoch, minibatch_index + 1,
                      this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    #test_losses = [test_model(i) for i
                    #               in [29]]
                    #test_score = numpy.mean(test_losses)
                    test_score, test_pred, test_actual = test_model(29)

                    print(('     epoch %i, minibatch %i, test error of '
                           'best model %f') %
                          (epoch, minibatch_index + 1,
                           numpy.sum(test_score)))

            if patience <= iter:
                    done_looping = True
                    break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f'
           'obtained at iteration %i, with test performance %f') %
          (best_validation_loss, best_iter + 1, numpy.sum(test_score)))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
	
    # store data

    #f = file('joint_fit_results/a2_params_all_neurons_s200' + '_lam_' + str(L1_reg) + '.save', 'wb')
    # file doesn't exist in python 3, use open instead
    f = open('joint_fit_results/a2_params_all_neurons_s200' + '_lam_' + str(L1_reg) + '.save', 'wb')
    #f = file('joint_fit_results/params_all_neurons' + '_lam_' + str(L1_reg) + '.save', 'wb')
    for obj in [params + [test_score] + [test_pred] + [test_actual]]:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    #plot the predicted trace
    #plt.clf()
    #plt.subplot(2,1,1)
    #plt.subplot(2, 1, 1)
    #plt.plot(test_pred, 'k')
    #plt.title('test data prediction')
    #plt.ylabel('predicted rate')
    #plt.subplot(2, 1, 2)
    #plt.plot(test_actual, 'k')
    #plt.xlabel('timebins')
    #plt.ylabel('actual spikes')
    #plt.savefig('single_fit_results/trace_n_' + str(n_index) + '_lam_' + str(L1_reg) + '.png')

    #plot the params and then show
    #plt.clf()
    #vis = plt.imshow(numpy.reshape(Layer0.W.eval(),(60,10),order='F'))
    ##tmp = numpy.max(numpy.abs(params[0].eval()))
    ##vis.set_clim(-tmp,tmp)
    #plt.colorbar()
    #plt.savefig('single_fit_results/RF_n_' + str(n_index) + '_lam_' + str(L1_reg) + '.png')
    #print numpy.max(numpy.abs(params[0].eval()))    
    #print numpy.mean(numpy.abs(params[0].eval()))
    #print numpy.median(numpy.abs(params[0].eval()))

#loop over neurons and cross validation parameters
if __name__ == '__main__':
    SGD_training(L1_reg=0)





