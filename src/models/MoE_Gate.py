import torch
import torch.nn as nn
from models.resnet1d import ResNet1d

import sys
sys.path.append('..')


class MoeGate(nn.Module):
    """
    Mixture of Experts (MoE) model using a pre-trained gate and three expert networks.
    
    Parameters
    ----------
    gate_path : str
        Path to the saved model for the gate network.
    experts_config : object
         Configuration class for the experts.
    num_classes : int
        Number of output classes for the model.
    """

    def __init__(self, gate_path, experts_config, num_classes):
        super().__init__()

        self.num_classes = num_classes
        # Load the pre-trained gate model from the specified path
        self.gate = torch.load(gate_path)
        
        # Initialize the experts as a ModuleList (container for holding multiple layers)
        self.experts = nn.ModuleList()
        for i in range(3):
            # Initialize each expert using the configuration provided by `experts_config`
            model = ResNet1d(**experts_config().__dict__)
            
            # Load the state dict (weights) from the gate model
            state_dict = torch.load(gate_path).state_dict()
            
            # Remove the linear layer weights from the gate model (since experts have their own)
            del state_dict['lin.weight']
            del state_dict['lin.bias']
            
            # Load the state dict into the expert model, without affecting the linear layer
            model.load_state_dict(state_dict, strict=False)
            
            # Append the expert model to the experts list
            self.experts.append(model)

    def forward(self, x):
        """
        Forward pass for the MoE model. Combines outputs from the gate and experts.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output logits after mixing the expert predictions using the gate.
        """
        # Get gate predictions for input x
        g = self.gate.forward(x)
        g = torch.sigmoid(g)  # Apply sigmoid to ensure gate outputs are between 0 and 1

        # Get the logits (predictions) from each expert
        logits = [expert.forward(x) for expert in self.experts]

        # Expand gate tensor to match the shape of logits for multiplication
        g = g.unsqueeze(1)
        g = g.expand(-1, self.num_classes, -1)

        # Combine expert outputs using gate values as weights
        logits = torch.stack(logits, dim=2)
        logits = torch.sum(g * logits, dim=2)

        return logits
    