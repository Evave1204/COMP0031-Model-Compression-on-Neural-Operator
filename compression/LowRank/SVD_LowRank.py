from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Union, Tuple, Optional
from neuralop.models import FNO
from neuralop.losses import LpLoss, H1Loss
from neuralop.data.datasets import load_darcy_flow_small
import numpy as np
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training.training_state import load_training_state
import torch
import torch.nn as nn
from typing import Dict
from compression.base import CompressionTechnique
import copy

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List


class SVDLowRank:
    """
    Handles low-rank decomposition for:
    - Conv1d(kernel_size=1) → Replaced with two smaller Conv1d layers.
    - SpectralConv (DenseTensor) → Performs low-rank SVD on channel dimensions.
    """
    def __init__(self, model, rank_ratio=0.5, min_rank=4, max_rank=32):
        self.rank_ratio = rank_ratio  
        self.min_rank = min_rank     
        self.max_rank = max_rank      
        self.model = model
        self.original_params = 0
        self.compressed_params = 0
        self.compressed_layers = {}
    
    def _get_target_rank(self, weight: torch.Tensor) -> int:
        if weight.is_complex():
            U_real, S_real, _ = torch.linalg.svd(weight.real.float())
            U_imag, S_imag, _ = torch.linalg.svd(weight.imag.float())
            S = (S_real + S_imag) / 2
        else:
            _, S, _ = torch.linalg.svd(weight.float())
        energy = (S ** 2).cumsum(dim=0) / (S ** 2).sum()
        valid_indices = torch.where(energy <= self.rank_ratio)[0]
        rank = valid_indices.numel() + 1 if valid_indices.numel() > 0 else 1
        return max(min(rank, self.max_rank), self.min_rank)

    def compress_conv1d(self, layer, name):
        """
        Compress a Conv1d(kernel_size=1) layer using SVD decomposition.
        Replaces the layer with: Conv1d(in_channels, rank, 1) → ReLU → Conv1d(rank, out_channels, 1)
        """
        W = layer.weight.data  # [out_channels, in_channels, 1]
        out_channels, in_channels, kernel_size = W.shape
        
        if kernel_size != 1:
            raise ValueError(f"Layer {name} is Conv1d but kernel_size != 1. Skipping.")

        # Reshape to (out_channels, in_channels)
        W_2d = W.view(out_channels, in_channels)

        # Perform SVD
        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)

        # Truncate to rank
        rank = self._get_target_rank(W_2d)
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        # Create two smaller Conv1d layers
        conv1 = nn.Conv1d(in_channels, rank, kernel_size=1, bias=False)
        conv2 = nn.Conv1d(rank, out_channels, kernel_size=1, bias=layer.bias is not None)

        # Assign new weights
        conv1.weight.data = Vh.unsqueeze(2)              # [rank, in_channels, 1]
        conv2.weight.data = (U * S).unsqueeze(2)         # [out_channels, rank, 1]

        if layer.bias is not None:
            conv2.bias.data = layer.bias.data.clone()

        class CompressedSequential(nn.Sequential):
            def __init__(self, *args, **kwargs):
                super().__init__(*args)
                self.out_channels = kwargs.get("out_channels")
        seq = CompressedSequential(conv1, conv2, nn.ReLU(), out_channels=out_channels)
        self.compressed_layers[name] = seq

    def compress_spectral_conv(self, layer, name):
        """
        Compress a SpectralConv layer by applying low-rank SVD on channel dimensions.
        """
        if not hasattr(layer.weight, "to_tensor"):
            print(f"[Warning] DenseTensor at {name} has no 'to_tensor()' method. Skipping.")
            return

        # Extract tensor
        W_torch = layer.weight.to_tensor()  # Shape: [C_out, C_in, Nx, Ny]
        C_out, C_in, Nx, Ny = W_torch.shape

        # Reshape to (C_out*C_in, Nx*Ny) and perform SVD
        W_2d = W_torch.reshape(C_out * C_in, Nx * Ny)

        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)

        # Truncate
        rank = self._get_target_rank(W_2d)
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        # Low-rank approximation
        U_S = U * S.unsqueeze(0)
        W_2d_low = U_S @ Vh
        W_low = W_2d_low.view(C_out, C_in, Nx, Ny)

        new_layer = copy.deepcopy(layer)
        # Store new tensor
        if hasattr(layer.weight, "from_tensor"):
            new_layer.weight.from_tensor(W_low)
            self.compressed_layers[name] = new_layer
        else:
            print(f"[Warning] DenseTensor at {name} has no 'from_tensor()' method. Skipping.")

    def compress(self):
        """
        Iterates through the model and applies low-rank decomposition.
        - Conv1d(kernel_size=1) is replaced with two smaller Conv1d layers.
        - SpectralConv (DenseTensor) is approximated using low-rank SVD.
        """
        self.original_params = sum(p.numel() for p in self.model.parameters())

        for name, module in self.model.named_modules():
            # Handle Conv1d (kernel_size=1)
            if isinstance(module, nn.Conv1d) and module.kernel_size == (1,):
                self.compress_conv1d(module, name)
            # Handle SpectralConv (DenseTensor)
            elif "SpectralConv" in type(module).__name__:
                if hasattr(module, "weight"):
                    self.compress_spectral_conv(module, name)

        # Replace original layers with compressed versions
        for name, new_layer in self.compressed_layers.items():
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = self.model
            if parent_name:
                for part in parent_name.split("."):
                    parent_module = getattr(parent_module, part)
            setattr(parent_module, child_name, new_layer)

        self.compressed_params = sum(p.numel() for p in self.model.parameters())

        print("[LowRank] Compression applied successfully.")
        print("Original:",self.original_params)
        print("Compressed", self.compressed_params)
        return self.model

    def get_compression_stats(self) -> Dict[str, Union[int, float]]:
        # Compute compression ratio & sparsity
        compression_ratio = self.compressed_params / self.original_params
        sparsity = 1 - compression_ratio        
        return {
            "original_parameters": self.original_params,
            "compressed_parameters": self.compressed_params,
            "compression_ratio": compression_ratio,
            "sparsity": sparsity,
            "compressed_layers": list(self.compressed_layers.keys())
        }

    


    # def _is_compressible(self, layer: nn.Module) -> bool:
    #     """Check if layer is a compressible linear layer"""
    #     return isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d))
    
    # def _get_target_rank(self, weight: torch.tensor) -> int:
    #     # S includes all eigenvalues, S = [σ₁, σ₂, ..., σₙ] where σ₁ ≥ σ₂ ≥ ... ≥ σₙ
    #     _, S, _ = torch.svd(weight.float())
    #     # E = total_energy = σ₁^2 + σ₂^2 + ... + σₙ^2
    #     # energy = the energy of each eigenvalue = [σ₁/E, σ₂/E, ..., σₙ/E]
    #     energy = (S**2).cumsum(dim=0) / (S ** 2).sum()
    #     # take suitable rank index : energy < rank ratio
    #     index = torch.where(energy <= self.rank_ratio)[0]
    #     # if no suitable indexs -> new rank = 1, where the eigenvalue is σ₁
    #     if len(index) == 0:
    #         new_rank = 1
    #     # if yes, new rank = max_index + 1, the index is continuous ... [0,1] | [0,1,2,..,max]
    #     else:
    #         new_rank = index.max().item() + 1
    #     # make sure the final rank >= min_rank, we set
    #     return max(new_rank, self.min_rank)
        
    # def _compress_linear(self, layer, name):
    #     weight = layer.weight.data
    #     U,S,V = torch.svd(weight.float())
    #     target_rank = self._get_target_rank(weight=weight)
    #     # compress W to be U@S,V
    #     self.compressed_layers[name] = (
    #         U[:, :target_rank] @ torch.diag(S[:target_rank]),
    #         V[:, :target_rank]
    #     )
    #     # Repalce the weight and bias
    #     layer.weight = nn.Parameter(self.compressed_layers[name][0])
    #     layer.bias = nn.Parameter(layer.bias.data) if layer.bias else None
    # def _compress_conv2d(self, layer, name):
    #     # weight = [out_channel, in_channel, H, W]
    #     weight = layer.weight.data
    #     # flatten weight = [out_channel, in_channel*H*W]
    #     flatten_weight = weight.view(weight.size(0), -1)
    #     # U = [out_channel, out_channel]
    #     # S = [min(out_channels), in_channel * H * W]
    #     # V = [in_channel * H * W, in_channel * H * W]
    #     U,S,V = torch.svd(flatten_weight.float())
    #     target_rank = self._get_target_rank(weight=flatten_weight)
    #     # U_rank = [out_channel, target_rank]
    #     # V_rank = [in_channel * H * W, target_rank]
    #     U_rank = U[:, :target_rank] @ torch.diag(S[:target_rank])
    #     V_rank = V[:, :target_rank]
    #     # reshape U_rank => U_rank= [out_chanenel, target_rank, 1, 1]
    #     U_rank = U_rank.unsqueeze(-1).unsqueeze(-1)
    #     # reshape V_rank => V_rank = [target_rank, in_channel, H, W]
    #     V_rank = V_rank.view(target_rank, weight.size(1), weight.size(2), weight.size(3))

    #     # build 2 conv2d layers
    #     self.compressed_layers[name] = (
    #         # by first conv2d: 
    #         # input_size(batch_size, weight_size(1)=input_channel, h, w) 
    #         # => output_size(batch_size, target_rank, h, w) 
    #         nn.Conv2d(weight.size(1), target_rank, kernel_size=1, bias=False),
    #         # by second conv2d:
    #         # input_size(batch_size, target_rank, h, w)
    #         # => output_size(batch_size, weight_size(0)=output_channel, O_h, O_w)
    #         nn.Conv2d(target_rank, weight.size(0), kernel_size=layer.kernel_size,
    #                   stride=layer.stride, padding=layer.padding, bias=True)
    #     )
    #     self.compressed_layers[name][0].weight = nn.Parameter(V_rank)
    #     self.compressed_layers[name][1].weight = nn.Parameter(U_rank)
    #     self.compressed_layers[name][1].bias = nn.Parameter(layer.bias.data) if layer.bias else None