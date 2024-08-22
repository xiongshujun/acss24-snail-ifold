"""
REPRESENTATION CAPACITY OF MLP OVER TRAINING TIME

Cohen et al (2020) shows that manifold separability improves as the depth of the network increases.
We make the argument that this improves over time during training as well

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

        For each y_i, create a matrix M_i, where the columns of M_i are made of the elements of X_i
        CSHL-2022 github code takes in a python list of all matrices M_i and spits out the capacity

        Let c(t) be the function of capacity over training epoch t
        We are interested in the behavior of c(t) with respect to more classical metrics in machine learning (e.g. MSE loss)

    Evaluation metrics
        Track c(t) and MSE loss over time

        Evaluate c(t) on
            Training data
            Evaluation data
            Test data 

"""

########################################
#                MODELS                #
########################################




########################################
#               DATASETS               #
########################################




########################################
#              EVALUATION              #
#               FUNCTION               #
########################################



########################################
#              EXPERIMENT              #
########################################

