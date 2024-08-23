"""
REPRESENTATION CAPACITY OF MLP OVER TRAINING TIME

Cohen et al (2020) shows that manifold separability improves as the depth of the network increases.
We make the argument that this improves over time during training as well

Stephenson and Padhy et al (2021) gives us four snapshots comparing manifold capacity demonstrating maximal capacity during generalization
    and associated phenomena (e.g. double descent, comparisons with loss over time), but still doesn't give us capacity and radii of mflds over time
We aim to show that this is the case as well. This paper also adds noise by randomly permuting labels, but we will also add noise within the input space itself (more input statistics, likely will increase the radii)

Lastly, we aim to look at capacity in cognitive tasks (simulated in NeuroGym) and, as a long shot, verification in vivo

Models:
    Linear 1-layer MLP with classifier on embedding (output of MLP)
    Multilayer MLP with classifier at the end of the MLP
    Cohen et al's model

Experiments:
    Dataset
        Simulated data of two (or more) Gaussian distributions
        Vectorized MNIST data
        NeuroGym tasks?

    Training
        Train MLP with respect to some classifier on top of some MLP output
            input ----> MLP --> embedding representation --> classifier --> classification to be learned over

Evaluation:
    How do we find manifold separability?
        Inputs z get mapped onto embeddings x by the weights w_MLP
            Let the set of all inputs with label y_i be Z_i = {z_{i_1}, z_{i_2}, ..., z_{i_k}}.
            Let the set of all embeddings corresponding to inputs Z_i be X_i,
                where x_{i_{j}} = f(w_MLP, z_{i_{j}}), 1 <= j <= k
                
            Hidden representations can be extracted using activation_extractor.py from the 2022 CSHL utils folder

        For each y_i, create a matrix M_i, where the columns of M_i are made of the elements of X_i
        CSHL-2022 github code takes in a python list of all matrices M_i and spits out the capacity
            This function gets manifold_geometry.manifold_analysis import manifold_analysis
        
        Let c(t) be the function of capacity over training epoch t
        We are interested in the behavior of c(t) with respect to more classical metrics in machine learning (e.g. MSE loss)

    Evaluation metrics
        Track c(t) and MSE loss over time

        Evaluate c(t) on
            Training data
            Evaluation data
            Test data 

"""
import torch
from torch import nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.optim import SGD

import numpy as np
import pickle

import time

from utils.manifold_analysis import *
from utils.activation_extractor import *
from utils.make_manifold_data import  *

def data_pickle(data, file_name, dir = 'representations_data\\'):
    with open(dir + file_name + '.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

########################################
#                MODELS                #
########################################

class SingleMLP(nn.Module):

    def __init__(self, input_size = 784, hidden_size = 100, output_size = 10):

        # input_size  := length of the flattened input
        # hidden_size := desired size of the embedded representation
        # output_size := desired size of output -- building a classifier over the task

        super(SingleMLP, self).__init__()   

        self.flatten = nn.Flatten()     

        self.learn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, output_size),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.learn(x)
        return logits
    
    def get_layers(self):
        return 2

class MultipleMLP(nn.Module):

    def __init__(self, input_size, output_size = 2):

        # input_size  := length of the flattened input
        # output_size := desired size of output -- building a classifier over the task

        super(MultipleMLP, self).__init__()      

        self.flatten = nn.Flatten()  

        self.learn = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.learn(x)
        return logits
    
    def get_layers(self):
        return 4

class PaperModel(nn.Module):

    def __init__(self):

        super(PaperModel, self).__init__()        

        pass ##!TODO: need to find implementation

    def forward(self, x):
        x = self.flatten(x)
        logits = self.learn(x)
        return logits
    
########################################
#               DATASETS               #
########################################

#!TODO: Simulated Gaussians
"""vector_size = 28*28 # same as MNIST
num_samples = 1000
class_num   = 2 # maximum number of classes
                    # for each class we draw a sample from a different Gaussian distribution 

data = []
for i in range(num_samples):
    data.append([])
# create Gaussians
random.randint(0, class_num - 1)"""

# MNIST data
mnist_training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

mnist_test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# generic dataloading function
def load(training_data, test_data, batch_size = 64, shuffle = True):

    train_dataloader = DataLoader(training_data, batch_size, shuffle)
    test_dataloader = DataLoader(test_data, batch_size, shuffle)

    return train_dataloader, test_dataloader

# manifold data simulation
def load_mfld(raw_data, sampled_classes = 10, examples_per_class = 40):
    data = make_manifold_data(raw_data, sampled_classes, examples_per_class) # Given the data, the number of classes, and the number of examples per class, we get the data we want as input to the neural network
    data = [d for d in data]
    return data

########################################
#              EVALUATION              #
#               FUNCTION               #
########################################

# Code adapted from CSHL 2022 codebase
def get_capacity(model, data, epoch, kappa = 0, n_t = 300):

    """
    One thing to note is that the CSHL 2022 code evaluates activations on training data. 
        This function refactors this to accept data (train or test) more generally
    
    The CSHL 2022 implementation gets snapshots of what the capacity metrics look like
        In the training loop, we will get a collection of the values at each layer over a given model and build their respective functions over time
    
    INPUT
        model := the model we are evaluating
        data  := the dataset over which this model will be evaluated
        epoch := the timepoint in training in which this model will be evaluated
        kappa := margin of capacity
        n_t   := number of Gaussian vectors to sample over

    OUTPUT
        layers := list of strs
            layers[i] := name of the i'th layer
        capacities:= a list of lists
            capacities[i] := [epoch, capacity, radius, dimension] for the i'th layer of activations
    """    

    activations_dict = extractor(model, data, layer_nums=[model.get_layers()]) # variability in number of layers per model


    # Reshape the manifold data to the expected dimensions
    for layer, activations in activations_dict.items():
        X = [d.reshape(d.shape[0],-1).T for d in activations]
        activations_dict[layer] = X
    
    # get the relevant variables more generally
    capacities = []
    layers = []
    for layer, activations in activations_dict.items():
        alpha, radius, dimension = manifold_analysis(X, kappa, n_t)
        c = 1/np.mean(1/alpha)
        r = np.mean(radius)
        d = np.mean(dimension)
        
        """
        print(f"{layer}, Capacity: {c}, Radius: {r}, Dimension: {d}")

        This print would return something like

        layer_0_Input, Capacity: 0.10210382868414464, Radius: 1.165534398073706, Dimension: 16.234330064336838

        or

        layer_3_ReLU, Capacity: 0. 10214180034859942, Radius: 1.1709662436924018, Dimension: 16.25642020167163
        """

        capacities.append([epoch, c, r, d])
        layers.append(layer)

    return layers, capacities

########################################
#              EXPERIMENT              #
########################################
# Define hyparameters, loss function, and optimizer

lr = 0.01            # learning rate
epochs = 20          # training time
batch_size = 64      

def eval(model, training_data, test_data, loss_fn = nn.CrossEntropyLoss()):

    """
    INPUT

    OUTPUT
        losses := list of 2-element lists, tracking loss over time
            losses[t] := loss at epoch t

        correct := float, test accuracy

        capacities_over_time := dict of dicts
            capacities_over_time['train_capacity_outputs'] := dict where the keys are a layer and the values form a list of [t, c(t)] pairs 
                                                                representing the capacity of the layer representations of training data over time
            capacities_over_time['train_radius_outputs'] := dict where the keys are a layer and the values form a list of [t, r(t)] pairs 
                                                                representing the radius of the layer representations of training data over time
            capacities_over_time['train_dimension_outputs'] := dict where the keys are a layer and the values form a list of [t, d(t)] pairs 
                                                                representing the dimension of the layer representations of training data over time
    
            capacities_over_time['test_capacity_outputs'] := dict where the keys are a layer and the values form a list of [t, c(t)] pairs 
                                                                representing the capacity of the layer representations of test data over time
            capacities_over_time['test_radius_outputs'] := dict where the keys are a layer and the values form a list of [t, r(t)] pairs 
                                                                representing the radius of the layer representations of test data over time
            capacities_over_time['test_dimension_outputs'] := dict where the keys are a layer and the values form a list of [t, d(t)] pairs 
                                                                representing the dimension of the layer representations of test data over time
    """

    optimizer = SGD(model.parameters(), lr)
    training_dataloader, test_dataloader = load(training_data, test_data)
    mfld_training_data = load_mfld(training_data)
    mfld_testing_data  = load_mfld(test_data)

    losses = []
    
    # TRAINING
    t = 0
    while t < epochs:

        l = 0 # keep track of total loss over an epoch
        for X, Y in training_dataloader:

            # prediction and loss
            pred = model(X)
            loss = loss_fn(pred, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l += loss

        # evaluation
        l = l.detach().item()
        losses.append([t, l/len(X)])
        print(f"Epoch: {t}, Loss: {loss:4f}")

        # get the capacity of each layer at this time point
            #!RECALL: capacities[i] := [epoch, capacity, radius, dimension] for the i'th layer of activations
        train_layers, train_capacities = get_capacity(model = model, data = mfld_training_data, epoch = t)
        test_layers, test_capacities   = get_capacity(model = model, data = mfld_testing_data, epoch = t)

        if t == 0: # case where we haven't set up the trackers of capacity
            train_capacity_outputs  = {}
            train_radius_outputs    = {}
            train_dimension_outputs = {}
            
            test_capacity_outputs  = {}
            test_radius_outputs    = {}
            test_dimension_outputs = {}
                # t_capacity_output[i] = [[t, c_i(t)]]
                # t_radius_output[i] = [[t, r_i(t)]]
                # t_dimension_output[i] = [[t, d_i(t)]]

            for i in range(len(train_capacities)):
                train_capacity_outputs[train_layers[i]]  = [[train_capacities[i][0], train_capacities[i][1]]]
                train_radius_outputs[train_layers[i]]    = [[train_capacities[i][0], train_capacities[i][2]]]
                train_dimension_outputs[train_layers[i]] = [[train_capacities[i][0], train_capacities[i][3]]]
                
                test_capacity_outputs[test_layers[i]]  = [[test_capacities[i][0], test_capacities[i][1]]]
                test_radius_outputs[test_layers[i]]    = [[test_capacities[i][0], test_capacities[i][2]]]
                test_dimension_outputs[test_layers[i]] = [[test_capacities[i][0], test_capacities[i][3]]]

        else:
            for i in range(len(train_capacities)):
                train_capacity_outputs[train_layers[i]].append([train_capacities[i][0], train_capacities[i][1]])
                train_radius_outputs[train_layers[i]].append([train_capacities[i][0], train_capacities[i][2]])
                train_dimension_outputs[train_layers[i]].append([train_capacities[i][0], train_capacities[i][3]])
                
                test_capacity_outputs[test_layers[i]].append([test_capacities[i][0], test_capacities[i][1]])
                test_radius_outputs[test_layers[i]].append([test_capacities[i][0], test_capacities[i][2]])
                test_dimension_outputs[test_layers[i]].append([test_capacities[i][0], test_capacities[i][3]])
        
        t += 1

    capacities_over_time = {
        'train_capacity_outputs': train_capacity_outputs,
        'train_radius_outputs' : train_radius_outputs,
        'train_dimension_outputs' : train_dimension_outputs,

        'test_capacity_outputs' : test_capacity_outputs,
        'test_radius_outputs' : test_radius_outputs,
        'test_dimension_outputs' : test_dimension_outputs
    }

    # TESTING
    correct = 0
    for X, Y in test_dataloader:
        pred = model(X)
        correct += (pred.argmax(1) == Y).type(torch.float).sum().item()
    
    correct /= len(test_dataloader.dataset)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}% \n")

    return losses, correct, capacities_over_time
    
########################################
#                SAVING                #
########################################
model = SingleMLP()
losses, correct, capacities_over_time = eval(model, training_data = mnist_training_data, test_data = mnist_test_data)

now = time.strftime('%d_%m_%Y-%H_%M_%S')
data_pickle(losses, 'training_loss_over_time_' + now)
data_pickle(correct, 'correct_' + now)
data_pickle(capacities_over_time, 'capacities_over_time_' + now)