import torch
import torch.ao.quantization
import torch.nn as nn
from compression.base import CompressionTechnique
from typing import Dict
import math
from neuralop.layers.spectral_convolution import SpectralConv
from .quantised_classes import *
from .observer_classes import *
from torch.nn import Linear, Conv1d, GroupNorm

# need to dequantise on the fly, currently storing quantised tensor prevents inference
# need to add different levels of quantisation

class UniformQuantisation(CompressionTechnique):
    def __init__(self, model: nn.Module, num_bits: int = 8, num_calibration_runs: int = 1):
        super().__init__(model)
        self.model = model
        self.num_bits = num_bits
        self.num_calibration_runs = num_calibration_runs
        if math.log2(num_bits) % 1 != 0:
            raise ValueError("Number of bits must be a power of 2")
        if num_bits == 8 or num_bits == 32:
            self.type = (lambda bits: getattr(torch, f"qint{bits}", None))(self.num_bits)
        else:
            self.type = (lambda bits: getattr(torch, f"int{bits}", None))(self.num_bits)
        self.init_size = self.get_size()

    def compress(self) -> None:
        self._quantise_model()
    
    def get_compression_stats(self) -> Dict[str, float]:
        return {"compression_ratio": self.get_size()/self.init_size,
                "bits": self.num_bits,
                "original_size": self.init_size,
                "compressed_size": self.get_size(),
                "sparsity": 1-(self.get_size()/self.init_size),
                }

    def _quantise_model(self) -> None:
        self.model = self.model.cpu()
        self.model.eval()
        model_name = self.model._get_name()
        quantise_methods = {
            "FNO": self._quantise_fno,
            "DeepONet": self._quantise_deeponet,
            # Add other model-specific quantization methods here
        }
        quantise_method = quantise_methods.get(model_name)
        if quantise_method:
            quantise_method()
        else:
            raise ValueError(f"Quantization method for model {model_name} is not defined")
        
    def _quantise_fno(self) -> None:
        self.model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        custom_class_observers = {"float_to_observed_custom_module_class": {SpectralConv: SpectralConvObserverCompatible}}
        torch.ao.quantization.prepare(self.model, inplace=True, prepare_custom_config_dict=custom_class_observers)

        for i in range(self.num):
            input_tensor = torch.randn(1, 1, 16, 16)
            self.model(input_tensor)

        torch.ao.quantization.convert(self.model, inplace=True)

        self.model.forward = partial(quantised_fno_forward, self.model)
        self.model.fno_blocks.forward_with_postactivation = partial(quantised_forward_with_postactivation, self.model.fno_blocks)
        self.model.fno_blocks.forward_with_preactivation = partial(quantised_forward_with_preactivation, self.model.fno_blocks)

    def _quantise_deeponet(self) -> None:
        pass

    
    def determine_input_shape(self) -> list[torch.Size]:
        if self.model._get_name() == "DeepONet":
            return [torch.Size([1, 1, 128, 128]), torch.Size([1, 1, 128, 128])]
        elif self.model._get_name() == "FNO":
            return [torch.Size([1, 1, 16, 16])]
        elif self.model._get_name() == "CODANO":
            return [torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 32, 32])]
        else:
            print(self.model)
            exit()

    def _get_parent_module(self, module_name: str):
        """
        Helper function to get the parent module of a given module name.
        """
        module_names = module_name.split('.')
        parent_module = self.model
        for name in module_names[:-1]:
            parent_module = getattr(parent_module, name)
        return parent_module
    
    def get_size(self) -> float:
        total_size = 0
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        total_size = param_size + buffer_size
        return total_size