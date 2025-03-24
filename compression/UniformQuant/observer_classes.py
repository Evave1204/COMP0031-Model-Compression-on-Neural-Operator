from .quantised_classes import *

class FNOObserver(QuantizedFNO):
    def __init__(self, fno):
        super().__init__(fno)

    @classmethod
    def from_float(cls, new):
        return FNOObserver(new)

class SpectralConvObserverCompatible(QuantizedSpectralConv):
    def __init__(self, spectral_layer):
        super().__init__(spectral_layer)
    
    @classmethod
    def from_float(cls, new):
        return SpectralConvObserverCompatible(new)
    
class LinearObserver(QuantizedLinear):
    def __init__(self, module):
        super().__init__(module)
    
    @classmethod
    def from_float(cls, new):
        return LinearObserver(new)

class Conv1dObserver(QuantizedConv1d):
    def __init__(self, module):
        super().__init__(module)
    
    @classmethod
    def from_float(cls, new):
        return Conv1dObserver(new)
    
class GroupNormObserver(QuantisedGroupNorm):
    def __init__(self, group_norm):
        super().__init__(group_norm)
    
    @classmethod
    def from_float(cls, new):
        return GroupNormObserver(new)
    
class SoftGatingObserver(nn.Module):
    def __init__(self, soft_gating):
        super().__init__()
        self.soft_gating = soft_gating

    def forward(self, x : torch.Tensor):
        x.dequantize()
        self.soft_gating(x)
        return x
    
    @classmethod
    def from_float(cls, new):
        return SoftGatingObserver(new)
