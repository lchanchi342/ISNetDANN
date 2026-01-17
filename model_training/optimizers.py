#optimizers.py


import torch


def get_sgd_optim(network, lr, weight_decay, momentum, nesterov=False):
    return torch.optim.SGD(
        network.parameters(), #uses all the model parameters
        lr=lr, #learining rate -> represents the step size for each update
        weight_decay=weight_decay, #L2 regularization 
        momentum=momentum, #accelerates convergence by considering the previous gradient updates.
        nesterov=nesterov) #applies the gradient shifted one step ahead, producing a more ‘anticipatory’ update (sometimes improves convergence)

def get_adam_optim(network, lr, weight_decay): #adaptive learning-rate mechanisms and momentum-based updates
    return torch.optim.Adam(
        network.parameters(), #uses all the model parameters
        lr=lr, #used as the initial step size for parameter updates
        weight_decay=weight_decay) #L2 regularization 


get_optimizers = {
    "sgd": get_sgd_optim,
    "adam": get_adam_optim,
}





