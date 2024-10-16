# Standard Libraries
import os  # For saving models and handling paths
import logging  # For logging training process and information

# Third-Party Libraries
import torch  # Core PyTorch library for tensor operations and deep learning
import torch.nn as nn  # PyTorch's neural network module
from tqdm import tqdm  # For creating progress bars during training loops
import numpy as np  # For numerical operations

# Custom Libraries
import sys
sys.path.append('..')  # Add the parent directory to the system path for importing custom modules
import utils  # Import utility functions (e.g., plotting, saving)

class MoeTrainer:
    """
    A class for training the Mixture of Experts (MoE) model.

    Parameters
    ----------
    device : torch.device
        Device where the model will be trained (e.g., 'cuda' or 'cpu').
    epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay (L2 regularization) for the optimizer.
    loss_func : function
        Loss function to compute training loss.
    optim_func : function
        Optimizer function to update model weights.
    model_name : str
        Name of the model to be saved.
    path_save_best_model : str
        Directory to save the best model.
    path_save_ckp : str
        Directory to save model checkpoints.
    path_save_plot : str
        Directory to save training plots.
    model : torch.nn.Module
        The Mixture of Experts model to be trained.
    loader_config : dict
        Configuration for data loading (train/validation loaders).
    save_weights_interval : int
        Interval for saving model weights.
    signal_non_zero_start : int
        Starting index of non-zero values in the signal data.
    signal_crop_len : int
        Length of the signal to be cropped.
    process_signal : str
        Option to process the signal (e.g., "non_zero").
    use_constrain : bool
        Whether to use importance constraints during training.
    """

    def __init__(self,
                 device,
                 epochs,
                 learning_rate,
                 weight_decay,
                 loss_func,
                 optim_func,
                 model_name,
                 path_save_best_model,
                 path_save_ckp,
                 path_save_plot,
                 model,
                 loader_config,
                 save_weights_interval,
                 signal_non_zero_start,
                 signal_crop_len,
                 process_signal,
                 use_constrain):
        
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_func = loss_func
        self.optim_func = optim_func
        self.model_name = model_name
        self.path_save_best_model = path_save_best_model
        self.path_save_ckp = path_save_ckp
        self.path_save_plot = path_save_plot
        self.model = model
        self.loader_config = loader_config
        self.save_weights_interval = save_weights_interval
        self.signal_non_zero_start = signal_non_zero_start
        self.signal_crop_len = signal_crop_len
        self.process_signal = process_signal
        self.use_constrain = use_constrain

        # If multiple GPUs are available, use DataParallel
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        
        # Move the model to the specified device (e.g., CUDA)
        self.model = self.model.to(self.device)

    def save_models(self, ckp, t):
        """
        Saves the model's weights to the specified path.
        
        Parameters
        ----------
        ckp : bool
            If True, save as a checkpoint, otherwise save as the best model.
        t : int
            Epoch number (used in checkpoint naming).
        """
        if not ckp:
            save_path = os.path.join(self.path_save_best_model, self.model_name + '.pt')
        else:
            save_path = os.path.join(self.path_save_ckp, self.model_name + f"_{t}.pt")
            
        torch.save(self.model, save_path)

    def get_inputs(self, batch, apply="non_zero", device="cuda"):
        """
        Pre-processes the input data by cropping signals.
        
        Parameters
        ----------
        batch : torch.Tensor
            Input batch data.
        apply : str, optional
            Process the signal based on non-zero start (default is "non_zero").
        device : str, optional
            Device to move the processed data (default is "cuda").
        
        Returns
        -------
        torch.Tensor
            Processed and cropped data moved to the specified device.
        """
        # If the number of leads is greater than the signal length, transpose the batch
        if batch.shape[1] > batch.shape[2]:
            batch = batch.permute(0, 2, 1)

        B, n_leads, signal_len = batch.shape

        if apply == "non_zero":
            transformed_data = torch.zeros(B, n_leads, self.signal_crop_len)
            for b in range(B):
                start = self.signal_non_zero_start
                diff = signal_len - start
                if start > diff:
                    correction = start - diff
                    start -= correction
                end = start + self.signal_crop_len
                for l in range(n_leads):
                    transformed_data[b, l, :] = batch[b, l, start:end]
        else:
            transformed_data = batch

        return transformed_data.to(self.device)

    def run(self):
        """
        Main training loop for the MoE model. Handles training and validation for each epoch.
        """
        loader = None
        train_loader, val_loader = loader.get_train_dataloader(), loader.get_val_dataloader()
                
        loss_func = self.loss_func
        optim_func = self.optim_func(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best_loss = 1e12
        conv_train_losses = []
        conv_val_losses = []

        logging.info("\n\n ----- STARTING TRAINING -----\n\n")
        
        for t in range(self.epochs):
            train_loss = 0.0    
            val_loss = 0.0
            for img, _, label, _ in tqdm(train_loader, desc=f'TRAINING EPOCH {t}/{self.epochs-1}', dynamic_ncols=True, colour="BLUE"):
                
                img = self.get_inputs(img, apply=self.process_signal).to(self.device)
                label = label.to(self.device).float()
                
                optim_func.zero_grad()

                pred = self.model(img)
                loss = loss_func(pred, label)
                loss.backward()
                optim_func.step()
                
                train_loss += loss.item()
                
            train_loss = train_loss / len(train_loader)
            conv_train_losses.append(train_loss)

            if self.use_constrain:
                self.model.reset_importance_accum()
            
            with torch.no_grad():
                for img, _, label, _ in val_loader:
                    img = self.get_inputs(img, apply=self.process_signal).to(self.device)
                    label = label.to(self.device).float()
                    pred = self.model(img)
                    loss = loss_func(pred, label)
                    val_loss += loss.item()
                
            val_loss = val_loss / len(val_loader)
            conv_val_losses.append(val_loss)

            if self.use_constrain:
                self.model.reset_importance_accum()

            if best_loss > val_loss:
                best_loss = val_loss
                self.save_models(ckp=False, t=t)
                
            logging.info(f"Epoch: {t}\nTrain Loss: {train_loss}\nValidation Loss: {val_loss}\n")

            if t != 0:
                utils.plot_loss(conv_train_losses, conv_val_losses, t, self.model_name, self.path_save_plot)
