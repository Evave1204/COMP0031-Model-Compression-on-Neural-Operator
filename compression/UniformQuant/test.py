import torch
import torch.ao.quantization
from compression.utils.fno_util import optional_fno
from compression.UniformQuant.observer_classes import *
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.skip_connections import SoftGating
from torch.nn import Linear, Conv1d, GroupNorm
from neuralop.models.fno import FNO
from functools import partial
from compression.UniformQuant.quantised_forwards import *

# Instantiate the FNO model
fno_model, train_loader, test_loaders, data_processor = optional_fno(resolution="low")
device = torch.device('cpu') #('cuda' if torch.cuda.is_available() else 'cpu')
fno_model = fno_model.to(device)

# Prepare the model for quantization
fno_model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
custom_class_observers = {"float_to_observed_custom_module_class": {SpectralConv: SpectralConvObserverCompatible}}
torch.ao.quantization.prepare(fno_model, inplace=True, prepare_custom_config_dict=custom_class_observers)

# Calibrate the model with some dummy input
for i in range(2):
    input_tensor = torch.randn(1, 1, 16, 16)
    fno_model(input_tensor)

# Convert the model to a quantized version
torch.ao.quantization.convert(fno_model, inplace=True)

# Add QuantStub and DeQuantStub to the model
fno_model.forward = partial(quantised_fno_forward, fno_model)
fno_model.fno_blocks.forward_with_postactivation = partial(quantised_forward_with_postactivation, fno_model.fno_blocks)
fno_model.fno_blocks.forward_with_preactivation = partial(quantised_forward_with_preactivation, fno_model.fno_blocks)
print(fno_model)

# Quantize the input tensor
tensor = torch.randn(1, 1, 16, 16)
val_min = tensor.min().item()
val_max = tensor.max().item()
scale = (val_max - val_min) / 256.0
if scale == 0:
    scale = 1.0
quantized_tensor = torch.quantize_per_tensor(tensor, scale, 0, torch.qint8)
print(f"\033[91m{fno_model}\033[00m")

# Run the lifting method on the quantized tensor
output = fno_model(tensor)

# Print the output
print(output)
