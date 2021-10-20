import torch
import math
from torch.utils.data import TensorDataset, DataLoader

def test(model,test_x, test_y,batch_size = 1024):
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)

    model.eval()
    predictive_means, predictive_variances, test_lls = model.predict(test_loader)

    rmse = torch.mean(torch.pow(predictive_means.mean(0) - test_y, 2)).sqrt()
    print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")
    return rmse.item(),-test_lls.mean().item()

