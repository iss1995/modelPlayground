import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader

from gpytorch.mlls import DeepApproximateMLL,ExactMarginalLogLikelihood
from gpytorch.mlls import VariationalELBO

def relativeLoss(y_pred,y_target):
    return torch.mean( ((y_pred-y_target)/torch.mean(torch.abs(y_target)))**2 )

def regularizer(coef,weights):
    return coef*torch.linalg.norm(weights[:-1]) 

def train(model, train_x, train_y, gp_in = None, num_epochs = 10, num_samples = 10, optimizer_function = torch.optim.Adam, lr = 0.01, batch_size=1024):

    optimizer = optimizer_function([ {'params':model.parameters()} ], lr = lr )
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

    epochs_iter = range(num_epochs)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        print(f"\tEpoch {i+1}/{num_epochs}")
        minibatch_iter = train_loader
        for x_batch, y_batch in minibatch_iter:
            with gpytorch.settings.num_likelihood_samples(num_samples):
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()

                # minibatch_iter.set_postfix(loss=loss.item())

    return model

def trainMetamodel(model, train_x, train_y, gp_in = None, num_epochs = 10, num_samples = 10, optimizer_function = torch.optim.Adam, lr = 0.01, batch_size=1024):

    optimizer = optimizer_function([ {'params':model.parameters()} ], lr = lr )
    mll = VariationalELBO(model.likelihood, model, train_x.shape[-2])
    # mll = ExactMarginalLogLikelihood(model.likelihood, model, train_x.shape[-2])
    # mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

    epochs_iter = range(num_epochs)

    if gp_in is None:
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_dataset = TensorDataset(train_x, train_y, gp_in)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        print(f"\tEpoch {i+1}/{num_epochs}")
        minibatch_iter = train_loader
        for x_batch, y_batch, gp_in_batch in minibatch_iter:
            with gpytorch.settings.num_likelihood_samples(num_samples):
                optimizer.zero_grad()
                output = model(gp_in_batch.squeeze(),x_batch.squeeze())
                loss = relativeLoss(output, y_batch)
                # loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()

                # minibatch_iter.set_postfix(loss=loss.item())

    return model