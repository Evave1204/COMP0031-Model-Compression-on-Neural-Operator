import torch
import torch.nn as nn
from typing import Dict, Union, List

'''
# Helper function to compute the total memory footprint (in bytes) of a model.
def _get_model_size_in_bytes(model: nn.Module) -> int:
    total_size = 0
    # Sum the sizes of all parameters
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    # Sum the sizes of all buffers (e.g., our quantized int8 buffers)
    for buffer in model.buffers():
        total_size += buffer.nelement() * buffer.element_size()
    return total_size
'''
class QuantizedLinear(nn.Module):
    '''
    Purpose:
    This class wraps an existing nn.Linear layer to “simulate” quantization.
    Explanation:
    The constructor takes an nn.Linear layer.
    It stores the number of input features, output features, and whether the layer uses a bias.
    '''
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.has_bias = linear.bias is not None


        # Compute scale from weight
        '''
        Purpose:
        Compute a scaling factor used to normalize the weight values.
        Explanation:
        The scale is the maximum absolute value in the weight tensor.
        If all values are zero (scale equals 0), we set it to 1.0 to avoid division by zero.
        '''
        # Instead of scaling by max(abs(weight)), scale to int8 range
        int8_max = 127  # Max range for int8
        self.scale = linear.weight.data.abs().max().item() / int8_max
        if self.scale == 0:
            self.scale = 1.0
        


        # Quantize weight to int8 and register as buffer
        '''
        Purpose:
        Create a quantized version of the weight.
        Explanation:
        The original weight is divided by the scale, then rounded to the nearest integer.
        The result is clamped to the valid int8 range (-128 to 127) and converted to int8.
        It is stored as a buffer (using register_buffer) so that it does not require gradients.
        '''
        q_weight = (linear.weight.data / self.scale).round().clamp(-128, 127).to(torch.int8)
        self.register_buffer('q_weight', q_weight)

        '''
        Purpose:
        Quantize the bias (if present) in the same way as the weight.
        Explanation:
        Computes a separate scale for the bias.
        Quantizes and registers the bias as a buffer. If no bias exists, registers None.
        '''
        if self.has_bias:
            self.bias_scale = linear.bias.data.abs().max().item()
            if self.bias_scale == 0:
                self.bias_scale = 1.0
            q_bias = (linear.bias.data / self.bias_scale).round().clamp(-128, 127).to(torch.int8)
            self.register_buffer('q_bias', q_bias)
        else:
            self.register_buffer('q_bias', None)



    '''
    Purpose:
    During the forward pass, convert the stored int8 quantized buffers back to float.
    Explanation:
    The quantized weight (and bias) are converted to float and multiplied by the 
    respective scale to recover an approximation of the original values.
    The standard linear function is then applied using these dequantized weights.
    '''
    def forward(self, x):
        # Dequantize weight and bias on-the-fly
        weight = self.q_weight.float() * self.scale
        bias = self.q_bias.float() * self.bias_scale if self.has_bias else None
        return nn.functional.linear(x, weight, bias)



class QuantizedConv1d(nn.Module):
    '''
    Purpose:
    A wrapper for a Conv1d layer that is designed for 1*1 convolutions.
    Explanation:
    Checks that the kernel size is exactly 1; otherwise, quantization isn't supported.

    '''
    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        if conv.kernel_size != (1,):
            raise ValueError("Only Conv1d with kernel_size=1 is supported")


        '''
        Purpose:
        Copy over all relevant convolution parameters.
        Explanation:
        Stores attributes like in_channels, out_channels, stride, etc., from the original convolution.
        '''    
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.has_bias = conv.bias is not None

        '''
        Purpose:
        Quantize the convolution weight.
        Explanation:
        Similar to QuantizedLinear: computes a scale factor and quantizes the weight to int8, storing it as a buffer.
        '''
        int8_max = 127  # Max range for int8
        self.scale = conv.weight.data.abs().max().item() / int8_max
        if self.scale == 0:
            self.scale = 1.0
        q_weight = (conv.weight.data / self.scale).round().clamp(-128, 127).to(torch.int8)
        self.register_buffer('q_weight', q_weight)

        '''
        Purpose:
        Quantize and store the bias (if available) in a similar manner.
        '''
        if self.has_bias:
            self.bias_scale = conv.bias.data.abs().max().item()
            if self.bias_scale == 0:
                self.bias_scale = 1.0
            q_bias = (conv.bias.data / self.bias_scale).round().clamp(-128, 127).to(torch.int8)
            self.register_buffer('q_bias', q_bias)
        else:
            self.register_buffer('q_bias', None)
    


    '''
    Purpose:
    Dequantize the weight and bias during the forward pass.
    Explanation:
    Converts the stored int8 buffers back to float by multiplying with the scale.
    Performs the convolution using the dequantized values.
    '''
    def forward(self, x):
        weight = self.q_weight.float() * self.scale
        bias = self.q_bias.float() * self.bias_scale if self.has_bias else None
        return nn.functional.conv1d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)



'''
Purpose:
This class performs dynamic quantization on the model by replacing eligible layers with their quantized versions.
'''
class DynamicQuantization:
    """
    Custom dynamic quantization that replaces eligible layers (nn.Linear and Conv1d with kernel_size=1)
    with quantized wrappers. For SpectralConv layers, if the weights are complex we skip quantization.
    """
    def __init__(self, model: nn.Module):
        '''
        Explanation:
        Stores the model to be compressed.
        Initializes a dictionary to track which layers have been replaced with quantized wrappers.
        '''
        self.model = model
        self.compressed_layers = {}  # Maps layer name to quantized module


    '''
    Purpose:
    Replace an nn.Linear layer with a QuantizedLinear wrapper.
    Explanation:
    Creates the wrapper, stores it in compressed_layers, and returns it.
    '''
    def compress_FC(self, layer: nn.Linear, name: str):
        quantized_linear = QuantizedLinear(layer)
        self.compressed_layers[name] = quantized_linear
        return quantized_linear


    '''
    Purpose:
Replace a 1*1 nn.Conv1d layer with a QuantizedConv1d wrapper.
Explanation:
Similar to compress_FC but for Conv1d layers.

    '''
    def compress_conv1d(self, layer: nn.Conv1d, name: str):
        if layer.kernel_size != (1,):
            raise ValueError(f"Layer {name} is Conv1d but kernel_size != 1. Skipping quantization.")
        quantized_conv = QuantizedConv1d(layer)
        self.compressed_layers[name] = quantized_conv
        return quantized_conv
    

    '''
    Purpose:
Process a SpectralConv layer if possible.
Explanation:
First, check if the layer's weight can be converted to a dense tensor via to_tensor().
If the weight is complex, print a warning and skip quantization.
Otherwise, quantize the weight (using quantize_tensor), then immediately dequantize it (multiply back by the scale).
Replace the weight using from_tensor() if available and store an identifier in compressed_layers.
    '''
    def compress_spectral_conv(self, layer, name: str):
        # Check for the to_tensor/from_tensor interface
        if not hasattr(layer.weight, "to_tensor"):
            print(f"[Warning] SpectralConv layer {name} does not support to_tensor(). Skipping quantization.")
            return layer
        W = layer.weight.to_tensor()
        if W.is_complex():
            print(f"[Warning] SpectralConv layer {name} has complex weights; dynamic quantization not supported. Skipping quantization.")
            return layer
        # For simplicity, quantize and then immediately dequantize
        q_W, scale_W = self.quantize_tensor(W)
        new_W = q_W.float() * scale_W
        if hasattr(layer.weight, "from_tensor"):
            layer.weight.from_tensor(new_W)
            self.compressed_layers[name] = "SpectralConv"
        else:
            print(f"[Warning] SpectralConv layer {name} does not support from_tensor(). Skipping quantization.")
        return layer

    '''
    Purpose:
A helper function to quantize any given tensor.
Explanation:
Computes the maximum absolute value (scale).
Divides the tensor by the scale, rounds, clamps to the int8 range, and converts to int8.
Returns both the quantized tensor and the scale.
    '''
    def quantize_tensor(self, tensor: torch.Tensor) -> (torch.Tensor, float):
        scale = tensor.abs().max().item()
        if scale == 0:
            scale = 1.0
        q_tensor = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return q_tensor, scale
    


    '''
    Purpose:
The main method to apply dynamic quantization across the model.
Explanation:
Computes the total number of parameters (used later for stats).
Iterates over each module (using named_modules()).
Depending on the type of the module, calls the corresponding compression function.
Replaces the original module in the model with the quantized version using replace_module.
Returns the modified model.
    '''
    def compress(self) -> nn.Module:
        '''
        self.original_params = sum(p.numel() for p in self.model.parameters())
        '''
        # 1) Store float size (model before compression)
        self.size_before = self._get_model_size_in_bytes(self.model)

        # Iterate over model modules and replace eligible ones with quantized wrappers.
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                quantized_module = self.compress_FC(module, name)
                self.replace_module(name, quantized_module)
            elif isinstance(module, nn.Conv1d) and module.kernel_size == (1,):
                quantized_module = self.compress_conv1d(module, name)
                self.replace_module(name, quantized_module)
            elif "SpectralConv" in type(module).__name__:
                new_module = self.compress_spectral_conv(module, name)
                self.replace_module(name, new_module)
        print("------------------------------[Dynamic Quantization] Compression applied successfully------------------------------")        
        return self.model
        
    

    '''
    Purpose:
Replace a module in the model given its name (e.g., "fno_blocks.convs.0").
Explanation:
Splits the module name by '.' to traverse the model's attribute tree.
Retrieves the parent module and sets the attribute corresponding to the final part to the new module.
    '''
    def replace_module(self, module_name: str, new_module: nn.Module):
        parts = module_name.split('.')
        parent = self.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_module)
    

    '''
    Purpose:
Provide a summary of the compression results.
Explanation:
Computes the original model size in bytes (assuming each parameter is float32, i.e., 4 bytes).
Computes a hypothetical quantized size if parameters were stored as int8 (1 byte each).
Calculates a compression ratio and a simulated "sparsity" (1 - compression ratio).
Returns these values along with the names of the compressed layers.
    '''

    '''
    def get_compression_stats(self) -> Dict[str, Union[int, float, List[str]]]:
        # Assume float32: 4 bytes per parameter.
        original_size = self.original_params * 4
        # Hypothetical quantized size if stored as int8: 1 byte per parameter.
        quantized_size = self.original_params * 1
        compression_ratio = quantized_size / original_size
        sparsity = 1 - compression_ratio  # Note: this is only a simulated value.
        return {
            "original_size": original_size,
            "quantized_size": quantized_size,
            "compression_ratio": compression_ratio,
            "sparsity": sparsity,
            "compressed_layers": list(self.compressed_layers.keys()),
        }
    '''

    def get_compression_stats(self):
        # 3) After compression, measure again
        size_after = self._get_model_size_in_bytes(self.model)
        ratio = size_after / self.size_before
        return {
            "original_size": self.size_before,
            "quantized_size": size_after,
            "compression_ratio": ratio,
            "sparsity": 1 - ratio,
            "compressed_layers": list(self.compressed_layers.keys()),
        }

    def _get_model_size_in_bytes(self, model):
        total_size = 0
        for p in model.parameters():
            total_size += p.nelement() * p.element_size()
        for b in model.buffers():
            total_size += b.nelement() * b.element_size()
        return total_size

