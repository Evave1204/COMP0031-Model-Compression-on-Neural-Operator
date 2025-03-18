import torch
import torch.nn as nn
from compression.base import CompressionTechnique
from typing import Dict

# need to dequantise on the fly, currently storing quantised tensor prevents inference
# need to add different levels of quantisation

class UniformQuantisation(CompressionTechnique):
    def __init__(self, model: nn.Module, num_bits: int = 8):
        super().__init__(model)
        self.model = model
        self.num_bits = num_bits
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
        self.model.eval()
        for name, param in self.model.named_parameters():
            if param.is_complex():
                continue
            scale = (param.abs().max().item() - param.min().item()) / torch.iinfo(self.type).max
            if scale == 0:
                scale = 1.0
            zero_point = 0
            x_q = torch.quantize_per_tensor(param, scale=scale, zero_point=zero_point, dtype=torch.qint8)
            x_q.requires_grad = False
            
            module, param_name = get_nested_attr(self.model, name)
            setattr(module, param_name, torch.nn.Parameter(x_q.dequantize(), requires_grad=False))
    
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

def get_nested_attr(obj, attr_str):
    print(str(obj), attr_str)
    parts = attr_str.split('.')
    attr = obj
    prev_attr = None
    for each in parts:
        if each.isdigit():
            try:
                attr = attr[int(each)]
            except: # except for when each is a number but attr is not a list (e.g dict)
                attr = getattr(attr, each)
        else:
            prev_attr = attr
            attr = getattr(attr, each)
    return prev_attr, parts[-1]