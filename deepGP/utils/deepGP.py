import torch

from utils.layers import ToyDeepGPHiddenLayer, GPHiddenLayer
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.models.deep_gps import DeepGP
from utils.train import relativeLoss

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
        hidden_rep1 = self.hidden_layer(inputs) # num_of_samples x (outputs x batch)
        output = self.last_layer(hidden_rep1) # how does this guy interprets the input?
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
    def __init__(self, gp_in_shape, num_output_dims, num_inducing=101, steps_ahead = 1):
        """
        Define the structure of your meta-model. The inducing points will be for all your feature models, the number of inputs should be the number of variables driving the GP ( e.g. T,height) and the number of outputs should be the number of your indipendent weights. Then the batch size should be the number of points on your grid.
        """
        last_layer = GPHiddenLayer(
            input_dims=gp_in_shape[-1], 
            output_dims=num_output_dims, # nop
            mean_type='constant',
            num_inducing=num_inducing,
        )

        super().__init__(gp_in_shape, num_output_dims)

        self.last_layer = last_layer
        self.likelihood = MultitaskGaussianLikelihood( num_tasks = num_output_dims)

    # TODO: extend to accept input for gp and regressor
    def forward(self, gp_inputs, regressor, predict = False ):
        """
        predict state evolution.
        dimensions:
        gp_inputs -> ip_vars x nop 
        weights -> unique_weights x nop  
        dynamics_vector -> reg_elements x nop  
        regressor -> reg_elements x nop
        responses -> 1 x nop 
        """
        weights = self.last_layer(gp_inputs) # num_of_samples x (outputs x batch)
        # depending on whether you are training and you sample weights or you validate
        mean_weights = weights.mean
        if not predict:
            dynamics_vector = self.dynVector(mean_weights)

            response_components = torch.mul(dynamics_vector ,regressor.repeat(dynamics_vector.shape[0],1,1))
        else:
            dynamics_vector = self.dynVector( torch.mean(mean_weights, axis=0) )

            response_components = torch.mul(dynamics_vector ,regressor)
        responses = torch.sum( response_components, dim=-1 )  
        return responses

    # TODO: extend for multiple output
    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch,gp_in in test_loader:
                preds = self(gp_in.squeeze(),x_batch.squeeze(),predict = True)
                mus.append(preds)

        return torch.hstack(mus)#, torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)

    def dynVector(self,x):
        if len(x.shape)>2:
            return torch.stack((
                                x[:,:,0],
                                x[:,:,1],x[:,:,1],x[:,:,1],x[:,:,1],
                                x[:,:,2],
                                x[:,:,2]*x[:,:,3],
                                x[:,:,4],
                                x[:,:,5]
                            ),axis = -1)
        else:
            return torch.stack((
                                x[:,0],
                                x[:,1],x[:,1],x[:,1],x[:,1],
                                x[:,2],
                                x[:,2]*x[:,3],
                                x[:,4],
                                x[:,5]
                            ),axis = -1) 