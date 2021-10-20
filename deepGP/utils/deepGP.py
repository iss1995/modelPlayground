import torch

from utils.layers import ToyDeepGPHiddenLayer
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGP

# architecture classes
class DeepGP(DeepGP):
    def __init__(self, train_x_shape,num_output_dims,num_inducing=128):
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='linear',
            num_inducing=num_inducing
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
            num_inducing=num_inducing
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)

# let's see
class GPMetaModel(DeepGP):
    def __init__(self, train_x_shape,num_output_dims,num_inducing=128):
        """
        Define the structure of your meta-model. The inducing points will be for all your feature models, the number of inputs should be the number of variables driving the GP ( e.g. T,height) and the number of outputs should be the number of your indipendent weights. Then the batch size should be the number of points on your grid.
        """
        last_layer = ToyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='constant',
            num_inducing=num_inducing
        )

        super().__init__()

        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    # TODO: extend to accept input for gp and regressor
    def forward(self, inputs):
        """
        predict state evolution
        """
        weights = self.last_layer(inputs)
        return weights

    # TODO: extend for multiple output
    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)