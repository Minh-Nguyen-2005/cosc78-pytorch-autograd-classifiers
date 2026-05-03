from torch import nn
from generalized_logistic_layer import GeneralizedLogisticLayer
from fully_connected_layer import FullyConnectedLayer


def create_net(input_features, hidden_units, non_linearity, output_size):
    """
    Constructs a network based on the specifications passed as input arguments

    Arguments
    --------
    input_features: (integer) the number of input features
    hidden_units: (list) of length L where L is the number of hidden layers. hidden_units[i] denotes the
                        number of units at hidden layer i + 1 for i  = 0, ..., L - 1
    non_linearity: (list)  of length L. non_linearity[i] contains a string describing the type of non-linearity to use
                           hidden layer i + 1 for i = 0, ... L-1
    output_size: (integer), the number of units in the output layer

    Returns
    -------
    net: (Sequential) the constructed model
    """

    # instantiate a sequential network
    net = nn.Sequential()

    # add the hidden layers
    in_features = input_features
    for i in range(len(hidden_units)):
        # Each hidden block is affine transformation followed by a chosen nonlinearity.
        net.add_module("fc_%d" % (i + 1), FullyConnectedLayer(in_features, hidden_units[i]))
        net.add_module("%s_%d" % (non_linearity[i], i + 1), GeneralizedLogisticLayer(non_linearity[i]))
        in_features = hidden_units[i]

    # add output layer
    # CrossEntropyLoss expects raw class scores, so no activation is added here.
    net.add_module("predictions", FullyConnectedLayer(in_features, output_size))

    return net
