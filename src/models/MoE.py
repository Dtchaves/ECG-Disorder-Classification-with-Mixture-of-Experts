import torch
import torch.nn as nn
from models.resnet1d import ResNet1d

import sys
sys.path.append('..')


class Moe(nn.Module):
    """
    Mixture of Experts (MoE) model using a ResNet1d gate and three expert networks.
    
    Parameters
    ----------
    gate_config : object
         Configuration class for the gate network.
    experts_config : object
       Configuration class for the experts.
    num_classes : int
        Number of output classes for the model.
    """

    def __init__(self, gate_config, experts_config, num_classes):
        super().__init__()

        self.num_classes = num_classes
        # Initialize the gate model using the configuration provided by `gate_config`
        self.gate = ResNet1d(**gate_config().__dict__)
        
        # Initialize the experts as a ModuleList (container for holding multiple layers)
        self.experts = nn.ModuleList()
        
        # Create three experts using the configuration from `experts_config`
        self.experts.append(ResNet1d(**experts_config().__dict__))
        self.experts.append(ResNet1d(**experts_config().__dict__))
        self.experts.append(ResNet1d(**experts_config().__dict__))

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
    