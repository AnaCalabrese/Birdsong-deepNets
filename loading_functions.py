"""
Collection of loading functions

"""
import numpy
import h5py
import theano

# This function is to load the data into the proper format.  
# This function loads data for all neurons.
def load_anas_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############

    print('... loading data')
    dataset='Data/MLd_prep_padded.mat'
    f            = h5py.File(dataset)
    responses    = numpy.transpose(f['responses'])
    stimuli      = numpy.transpose(f['stimulus'])
    maxBatchSize = numpy.array(f['maxBatchSize'])
    ntrials      = numpy.transpose(f['ntrials'])
    useMask      = numpy.transpose(f['useMask'])
    f.close()

    data_set = (stimuli, responses)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    data_set_x, data_set_y = shared_dataset(data_set)

    data_labels = (ntrials, useMask)
    y_counts, y_mask = shared_dataset(data_labels)

    rval = [(data_set_x, data_set_y), y_counts, y_mask, maxBatchSize]

    return rval

# This function is to load the data into the proper format.
# This function only will return data for 1 neuron at a time.
def load_anas_data_1(n_index):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############

    print('... loading data')
    dataset='Data/MLd_prep_padded.mat'
    #dataset='Data/GLM_test.mat'
    f = h5py.File(dataset)
    responses = numpy.transpose(f['responses'])
    stimuli = numpy.transpose(f['stimulus'])
    maxBatchSize = numpy.array(f['maxBatchSize'])
    ntrials = numpy.transpose(f['ntrials'])
    useMask = numpy.transpose(f['useMask'])
    f.close()

    responses = responses[:,n_index]
    ntrials = ntrials[:,n_index]
    useMask = useMask[:,n_index]

    data_set = (stimuli, responses)

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y

    data_set_x, data_set_y = shared_dataset(data_set)

    data_labels = (ntrials, useMask)
    y_counts, y_mask = shared_dataset(data_labels)

    rval = [(data_set_x, data_set_y), y_counts, y_mask, maxBatchSize]

    return rval


