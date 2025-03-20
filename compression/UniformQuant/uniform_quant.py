import torch
import torch.nn as nn
from compression.base import CompressionTechnique
from typing import Dict
import math
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.embeddings import GridEmbeddingND

# need to dequantise on the fly, currently storing quantised tensor prevents inference
# need to add different levels of quantisation

class UniformQuantisation(CompressionTechnique):
    def __init__(self, model: nn.Module, num_bits: int = 8):
        super().__init__(model)
        self.model = model
        self.num_bits = num_bits
        if math.log2(num_bits) % 1 != 0:
            raise ValueError("Number of bits must be a power of 2")
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
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        self.model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
            weight=torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
        )
        custom_class_observers = {} # {"float_to_observed_custom_module_class": {SpectralConv: QuantizedSpectralConv}}
        self.model = torch.quantization.prepare(self.model, inplace=True, prepare_custom_config_dict=custom_class_observers)
        tensor_sizes = self.determine_input_shape()
        for _ in range(1):
            input_tensors = [torch.randn(size) for size in tensor_sizes]
            for input_tensor in input_tensors:
                input_tensor = input_tensor.cpu()
            self.model(*input_tensors)
        self.model = torch.quantization.convert(self.model, inplace=True)
        print(f"\033[91m{self.model}\033[00m")
    
    def determine_input_shape(self) -> list[torch.Size]:
        if self.model._get_name() == "DeepONet":
            return [torch.Size([1, 1, 1, 1]), torch.Size([1, 1, 1, 1])]
        elif self.model._get_name() == "FNO":
            return [torch.Size([1, 1, 16, 16])]

            


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

class QuantizedSpectralConv(nn.Module):
    def __init__(self, spectral_conv: SpectralConv):
        super().__init__()
        self.spectral_conv = spectral_conv
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x, *args, **kwargs):
        x = self.quant(x)
        x = self.spectral_conv(x, *args, **kwargs)
        x = self.dequant(x)
        return x
    
    @classmethod
    def from_float(cls, float_module: SpectralConv):
        quant_module = cls(float_module)
        quant_module.quant = torch.quantization.QuantStub()
        quant_module.dequant = torch.quantization.DeQuantStub()
        return quant_module

    @classmethod
    def transform(cls, float_module: SpectralConv, output_shape=None, **kwargs):
        return cls.from_float(float_module)