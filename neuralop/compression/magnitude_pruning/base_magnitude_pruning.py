from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Union, Tuple, Optional
from neuralop.models import FNO

class BaseMagnitudePruning(ABC):
    """Base class for implementing pruning strategies"""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.masks: Dict[str, torch.Tensor] = {}
        
    @abstractmethod
    def compute_masks(self) -> Dict[str, torch.Tensor]:
        """Compute binary masks for parameters based on magnitude"""
        pass
        
    def apply_masks(self) -> None:
        """Apply stored masks to parameters"""
        for name, mask in self.masks.items():
            param = self.get_parameter(name)
            param.data *= mask
            
    def get_sparsity(self) -> float:
        """Calculate overall sparsity ratio"""
        total_params = 0
        zero_params = 0
        
        for mask in self.masks.values():
            total_params += mask.numel()
            zero_params += (mask == 0).sum().item()
            
        return zero_params / total_params if total_params > 0 else 0

class FNOPruning(BaseMagnitudePruning):
    """FNO-specific pruning implementation"""
    
    def __init__(self, model: nn.Module, threshold: float = 0.1):
        super().__init__(threshold)
        self.model = model
        
    def compute_masks(self) -> Dict[str, torch.Tensor]:
        """Compute masks for FNO parameters"""
        masks = {}
        
        # Prune spectral convolution weights
        for layer_idx, conv in enumerate(self.model.fno_blocks.convs):
            weight = conv.weight
            threshold = self.threshold * weight.abs().max()
            mask = (weight.abs() > threshold).float()
            masks[f'fno_blocks.convs.{layer_idx}.weight'] = mask
            
        # Prune lifting layer weights
        if hasattr(self.model.lifting, 'weight'):
            weight = self.model.lifting.weight
            threshold = self.threshold * weight.abs().max()
            mask = (weight.abs() > threshold).float()
            masks['lifting.weight'] = mask
            
        # Prune projection layer weights
        if hasattr(self.model.projection, 'weight'):
            weight = self.model.projection.weight
            threshold = self.threshold * weight.abs().max()
            mask = (weight.abs() > threshold).float()
            masks['projection.weight'] = mask
            
        self.masks = masks
        return masks

class CompressedModel(nn.Module):
    """Wrapper class that adds pruning capabilities to any model"""
    
    def __init__(self, 
                 model: nn.Module,
                 compress_class: type):
        super().__init__()
        self.model = model
        self.compress = compress_class(model)
        
    def compress_class(self) -> None:
        """Execute pruning"""
        self.compress()
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

fno_model = FNO(n_modes=(16, 16), hidden_channels=64, in_channels=3, out_channels=1)

pruned_model = CompressedModel(
    model=fno_model,
    compress_class=FNOPruning
)

pruned_model.compress()

# Get sparsity ratio
sparsity = pruned_model.get_sparsity()
print(f"Model sparsity: {sparsity:.2%}")