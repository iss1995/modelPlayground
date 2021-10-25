import torch

def lineInputs(noe = 1000):
    line = torch.linspace(0,1,noe)
    X = 0.05*torch.randn((noe, 80, 9)) # features
    for i in range(80):
        for j in range(9):
            X[:,i,j] += line

    line *= 2
    gp_in = 0.05*torch.randn((noe, 80, 2)) # gp variables
    for i in range(80):
        for j in range(2):
            gp_in[:,i,j] += line 

    y = 0.05*torch.randn((noe, 80)) # system states
    line *= 0.5
    for i in range(80):
        y[:,i] += line

    return X,y,gp_in

def oneInputs(noe = 1000):
    X = torch.ones((noe,80,9))
    y = torch.ones((noe,80))
    gp_in = torch.ones((noe,80,2))
    return X,y,gp_in