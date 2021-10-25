import torch

from math import floor
from utils.deepGP import GPMetaModel
from utils.train import trainMetamodel
from utils.test import dummyTest
from utils.helper import lineInputs,oneInputs
if __name__ == "__main__":

    # create data
    # X,y,gp_in = lineInputs()
    X,y,gp_in = oneInputs()
    gp_in += 0.1*torch.randn_like(gp_in)

    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()
    train_gp_in = gp_in[:train_n].contiguous()

    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()
    test_gp_in = gp_in[train_n:].contiguous()

    if torch.cuda.is_available():
        train_x, train_y, train_gp_in, test_x, test_y, test_gp_in = train_x.cuda(), train_y.cuda(), train_gp_in.cuda(), test_x.cuda(), test_y.cuda(), test_gp_in.cuda()

    num_epochs = 2
    num_samples = 10
    num_output_dims = 6 # number of independent weights


    model = GPMetaModel(train_gp_in.shape,num_output_dims=num_output_dims)
    if torch.cuda.is_available():
        model = model.cuda()

    print("Starting training...")
    model = trainMetamodel(
            model, train_x, train_y, gp_in = train_gp_in,
                                num_epochs = num_epochs, 
                                num_samples = num_samples, 
                                batch_size = 1 # you included everything in the point
                                )
    print("Done!")

    print("Validating...")
    # validate
    preds = dummyTest(model,test_x, test_y, test_gp_in, batch_size = 1)
    print("Done!")
    print("Finished dummy script")