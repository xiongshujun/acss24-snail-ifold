import pickle

import numpy as np
import matplotlib.pyplot as plt

from utils.pickling import *

########################################
#               PLOTTING               #
########################################

"""
!RECALL
    losses := list of 2-element lists, tracking loss over time
        losses[t] := loss at epoch t

    correct := float, test accuracy

    capacities_over_time := dict of dicts
        capacities_over_time['train_capacity_outputs']  := dict where the keys are a layer and the values form a list of [t, c(t)] pairs 
                                                            representing the capacity of the layer representations of training data over time
        capacities_over_time['train_radius_outputs']    := dict where the keys are a layer and the values form a list of [t, r(t)] pairs 
                                                            representing the radius of the layer representations of training data over time
        capacities_over_time['train_dimension_outputs'] := dict where the keys are a layer and the values form a list of [t, d(t)] pairs 
                                                            representing the dimension of the layer representations of training data over time

        capacities_over_time['test_capacity_outputs']   := dict where the keys are a layer and the values form a list of [t, c(t)] pairs 
                                                            representing the capacity of the layer representations of test data over time
        capacities_over_time['test_radius_outputs']     := dict where the keys are a layer and the values form a list of [t, r(t)] pairs 
                                                            representing the radius of the layer representations of test data over time
        capacities_over_time['test_dimension_outputs']  := dict where the keys are a layer and the values form a list of [t, d(t)] pairs 
                                                            representing the dimension of the layer representations of test data over time
"""

# Unpickle the data
losses = unpickle('training_loss_over_time_082324_1')
capacities_over_time = unpickle('capacities_over_time_082324_1')

parameters = list(capacities_over_time.keys()) # Get each of the 6 parameters we're looking at
layers     = list(capacities_over_time[parameters[0]].keys()) # Get the layer names of this model

fig_dir = 'figures\\'
suffix  = '082324_1.png'

# Losses
losses = np.array(losses).T
fig, ax = plt.subplots()
ax.plot(losses[0], losses[1])

ax.set(xlabel = 'Epoch', ylabel = 'Loss',
       title = 'Training loss with respect to Time')

fig.savefig(fig_dir + 'trainingloss_082324_1.png')
plt.show()

# Loop over each set of parameters and each combination of layers
# capacities_over_time[param][layer] := list of (t, f(t)) where f(t) is the parameter we're looking at with respect to time at the given layer
for param in parameters:
    for layer in layers:
        
        data = np.array(capacities_over_time[param][layer]).T
        fig, ax = plt.subplots()
        ax.plot(data[0], data[1])

        param_str = param[:-8]

        ax.set(xlabel = 'Epoch', ylabel = param_str,
            title = param_str + ' in ' + layer + ' with respect to Time')

        fig.savefig(fig_dir + param_str + '_' + layer + '_' + suffix)
        plt.show()

# Plot all pairs of [same layer - train/test] on the same graphs
# Since there are 3 metrics, all i with the same value modulo 3 contain the same parameters
for layer in layers:
    for j in range(3):
        fig, ax = plt.subplots()

        param = parameters[j]
        param_str = param[param.find('_') + 1: -8]

        for i in range(len(parameters)):            
            if i % 3 == j:

                param = parameters[i]
                condition = param[:param.find('_')]

                data = np.array(capacities_over_time[param][layer]).T
                ax.plot(data[0], data[1], label = condition)
        
        ax.set(xlabel = 'Epoch', ylabel = param_str,
            title = param_str + ' in ' + layer + ' with respect to Time')
        plt.legend(title='Setting:')

        fig.savefig(fig_dir + param_str + '_' + layer + '_' + suffix)
        plt.show()

# Plot all pairs of [same metric - different layers] on the same graph
for j in range(3):
    fig, ax = plt.subplots()

    param = parameters[j]
    param_str = param[param.find('_') + 1: -8] + ' Across Layers'

    for i in range(len(parameters)):            
        if i % 3 == j:

            param = parameters[i]
            condition = param[:param.find('_')]

            for layer in layers:

                data = np.array(capacities_over_time[param][layer]).T
                condition = param[:param.find('_')] + '_' + layer
                ax.plot(data[0], data[1], label = condition)

    ax.set(xlabel = 'Epoch', ylabel = param_str,
            title = param_str + ' with respect to Time')
    plt.legend(title='Setting:')

    fig.savefig(fig_dir + param_str + '_' + suffix)
    plt.show()

# Plot all the training stuff on the same graph


# Plot all the testing stuff on the same graph
