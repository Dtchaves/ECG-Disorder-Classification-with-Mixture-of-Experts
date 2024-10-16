from dataclasses import dataclass, field
from typing import Any

import torch 

import sys
sys.path.append('..')

from models.resnet1d import ResNet1d
from models.MoE import Moe
from models.MoE_Gate import MoeGate
from models.MoE_GateC import MoeGateC

from trainer.trainer import MoeTrainer

# ================================
# MODELS CONFIGURATION
# ================================

@dataclass
class ResNet1dConfig:
    blocks_dim: list = field(
        default_factory=lambda: [(64, 1024), (128, 256), (196, 64), (256, 16), (320, 4)]
    )
    n_classes: int = 6
    input_dim: int = (12, 4096)
    kernel_size: int = 17
    dropout_rate: int = 0.8
    
@dataclass
class GateConfig:
    blocks_dim: list = field(
        default_factory=lambda: [(64, 1024), (128, 256), (196, 64), (256, 16), (320, 4)]
    )
    n_classes: int = 3
    input_dim: int = (12, 4096)
    kernel_size: int = 17
    dropout_rate: int = 0.8
    
@dataclass
class MoeConfig:
    gate_config:GateConfig = field(default_factory=lambda:GateConfig)
    experts_config: ResNet1dConfig = field(default_factory=lambda:ResNet1dConfig)
    num_classes:int = 6 

@dataclass
class MoeGateConfig:
    gate_path:str = ""
    experts_config: ResNet1dConfig = field(default_factory=lambda:ResNet1dConfig)
    num_classes:int = 6 
    
@dataclass
class MoeGateCConfig:
    gate_path:str = ""
    experts_config: ResNet1dConfig = field(default_factory=lambda:ResNet1dConfig)
    num_classes:int = 6 
    importance_threshold:int = 0.5
    
    
# ================================
# TRAINER CONFIGURATION
# ================================

@dataclass
class TrainerConfig:
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs: int = 20
    learning_rate: float = 1e-6
    weight_decay:float = 0.01
    loss_func:  torch.nn.BCEWithLogitsLoss = field(default_factory=lambda:torch.nn.BCEWithLogitsLoss())
    optim_func:  torch.optim.AdamW = field(default_factory=lambda:torch.optim.AdamW)

    
    model_name:str = ""
    path_save_best_model: str = ''
    path_save_ckp: str = ''
    path_save_plot: str = ''
    
    model: None = None
    loader_config: None = None
    
    save_weights_interval: int = 10
    
    signal_non_zero_start: int = 571
    signal_crop_len: int = 2560
    process_signal:str = "normal"
    use_constrain:bool = True
    

# ================================
# MAIN CONFIGURATION
# ================================
    
@dataclass
class MainConfig:
    
    train: None = None
    train_bool: bool = False
    
    evaluate: None = None
    evaluate_bool: bool = False