# Script for plotting results from fitting the model
import pickle
import numpy
from pylab import *
import matplotlib.pyplot as plt

f = open('params_all_neurons_lam_0.save', 'rb')
loaded_object = pickle.load(f, encoding='latin1')
f.close()

""" load object contains 5 things:
    1. W matrix (loaded_object[0])
    2. b value (loaded_object[1])
    3. Array of negative log likelihood values for all cells on test data (loaded_object[2])
    4. Prediction for each neuron on test data (for one trial) (loaded_object[3])
    5. Summed spikes over all trials for each neuron (actually observed) (loaded_object[4])
"""
Wmat = loaded_object[0].eval()
print(numpy.shape(Wmat))

pred = loaded_object[3]
actual = loaded_object[4]

j = 39 # example cell to plot

plt.clf()
plt.subplot(2, 2, 1)
plt.plot(pred[:, j], 'k')
plt.title('test data prediction')
plt.ylabel('predicted rate')

plt.subplot(2, 2, 2)
plt.plot(actual[:, j], 'k')
plt.xlabel('time bins')
plt.ylabel('actual spikes')
plt.show()


# Why only 10 time bins? look at spike width
Wall = numpy.reshape(Wmat, (60, 10, 107), order='F')
plt.clf()
vis = plt.imshow(Wall[:, :, j])
plt.show()
