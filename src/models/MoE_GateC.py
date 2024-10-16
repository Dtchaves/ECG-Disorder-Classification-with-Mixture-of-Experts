import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('.')

from models.resnet1d import ResNet1d


class MoeGateC(nn.Module):
    """
    Mixture of Experts (MoE) model with three ResNet1d experts and a gating mechanism.
    
    Parameters
    ----------
    gate_path : str
        Path to the pre-trained gate model.
    experts_config : object
        Configuration class for initializing the experts (ResNet1d).
    num_classes : int
        Number of output classes for classification.
    importance_threshold : float
        Threshold for the importance constraint applied to expert gating.
    """

    def __init__(
        self, 
        gate_path,
        experts_config,
        num_classes,
        importance_threshold,
        ):
                
        super().__init__()

        # Number of output classes
        self.num_classes = num_classes
        
        # Load the pre-trained gate model from the provided path
        self.gate = torch.load(gate_path)

        # Initialize experts (3 ResNet1d models) using the provided configuration
        self.experts = nn.ModuleList()
        self.experts.append(ResNet1d(**experts_config().__dict__))
        self.experts.append(ResNet1d(**experts_config().__dict__))
        self.experts.append(ResNet1d(**experts_config().__dict__))

        # Accumulator to track the importance of each expert over time
        self.expert_importance_accum = np.zeros(3)  
        
        # Counter to track the number of forward passes
        self.count = 0
        
        # Threshold for applying the hard mean constraint to gating
        self.importance_threshold = importance_threshold

    def apply_hard_mean_constraint(self, g, x):
        """
        Apply a constraint to the gating values to enforce balanced expert utilization.
        
        Parameters
        ----------
        g : tensor
            Gating values indicating the importance of each expert.
        x : tensor
            Input data, used to normalize the constraint.
        
        Returns
        -------
        g_clone : tensor
            Modified gating values with the mean constraint applied.
        """
        g_clone = g.clone()  # Clone the gating tensor for manipulation
        line_x = x.size()[0]  # Batch size
        
        # Compute total and mean importance for the experts
        total_importance = np.sum(self.expert_importance_accum)
        mean_importance = total_importance / 3
        
        # Calculate the relative importance of each expert compared to the mean
        relative_importance = (self.expert_importance_accum - mean_importance) / mean_importance
        
        # Normalize the importance adjustment based on the number of forward passes
        mean_counstrain = (relative_importance / line_x) / self.count 
        
        # Subtract mean importance from the constraint
        mean_importance = mean_counstrain - mean_importance
        
        # Apply constraint: mask out gating values for over-utilized experts
        for i in range(3): 
            if mean_importance[i] > self.importance_threshold:
                g_clone[:, i].masked_fill_(g_clone[:, i] > 0, 0) 
        return g_clone

    def forward(self, x):
        """
        Forward pass for the MoE model.
        
        Parameters
        ----------
        x : tensor
            Input data.
        
        Returns
        -------
        logits : tensor
            Final output logits after mixing expert predictions based on gating.
        """
        # Compute gating values using the pre-trained gate model
        g = self.gate.forward(x)
        
        # Get predictions (logits) from each expert
        logits = [expert.forward(x) for expert in self.experts]
        
        # Update the importance accumulator for each expert based on the current gating
        self.expert_importance_accum += g.sum(dim=0).cpu().detach().numpy().reshape(-1)
        self.count += 1

        # Apply the hard mean constraint to the gating values
        g_clone = self.apply_hard_mean_constraint(g, x)
        
        # Expand gating values to match the dimensions of expert predictions
        g_clone = g_clone.unsqueeze(1)
        g_clone = g_clone.expand(-1, self.num_classes, -1)
        
        # Stack expert logits and weight them by the adjusted gating values
        logits = torch.stack(logits, dim=2)
        logits = torch.sum(g_clone * logits, dim=2)

        return logits

    def reset_importance_accum(self):
        """
        Reset the importance accumulator and count, used between training epochs.
        """
        self.expert_importance_accum = np.zeros(3)
        self.count = 0
