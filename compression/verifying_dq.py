#################### test_dynamic_quant.py ####################
import torch
import torch.nn as nn

# Adjust the import below to match the name of your own file/module
# where DynamicQuantization (and possibly QuantizedLinear, QuantizedConv1d) is defined.
from compression.quantization.dynamic_quantization import DynamicQuantization

def test_linear():
    """Test quantization of a simple Linear layer"""
    print("\n=================== LINEAR LAYER TEST ===================")
    
    # 1) Create a simple Linear model
    float_model = nn.Sequential(nn.Linear(in_features=4, out_features=1, bias=False))
    
    # 2) Manually set its weights to [1.11, 2.22, 3.33, 4.44]
    float_model[0].weight.data = torch.tensor(
        [[1.11, 2.22, 3.33, 4.44]], dtype=torch.float32
    )
    
    print("=== BEFORE COMPRESSION ===")
    print("Original float weight:", float_model[0].weight.data)
    
    # Quick forward pass in float mode
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    y_float = float_model(x)
    print("Float model output:", y_float)

    # 3) Apply dynamic quantization
    quantizer = DynamicQuantization(float_model)
    quantized_model = quantizer.compress()
    
    print("\n=== AFTER COMPRESSION ===")
    wrapper_layer = quantized_model[0]  # The wrapped QuantizedLinear
    
    print("Scale used for weight:", wrapper_layer.scale)
    print("Quantized weight (int8):", wrapper_layer.q_weight)
    
    dequantized_w = wrapper_layer.q_weight.float() * wrapper_layer.scale
    print("Dequantized weight:", dequantized_w)
    
    # 5) Forward pass with quantized model
    y_quant = quantized_model(x)
    print("Quantized model output:", y_quant)
    print("========================================================")


def test_conv1d():
    """Test quantization of a simple 1x1 Conv1d layer"""
    print("\n=================== CONV1D LAYER TEST ===================")

    # 1) Create a simple Conv1d model (1x1 kernel size)
    float_model = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, bias=False))

    # 2) Manually set its weights
    float_model[0].weight.data = torch.tensor([[[1.11, 2.22, 3.33, 4.44]]], dtype=torch.float32)  # Shape [out, in, kernel]

    print("=== BEFORE COMPRESSION ===")
    print("Original float weight:", float_model[0].weight.data)
    
    # Quick forward pass in float mode
    x = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32)  # Shape [batch, channels, width]
    y_float = float_model(x)
    print("Float model output:", y_float)

    # 3) Apply dynamic quantization
    quantizer = DynamicQuantization(float_model)
    quantized_model = quantizer.compress()
    
    print("\n=== AFTER COMPRESSION ===")
    wrapper_layer = quantized_model[0]  # The wrapped QuantizedConv1d
    
    print("Scale used for weight:", wrapper_layer.scale)
    print("Quantized weight (int8):", wrapper_layer.q_weight)
    
    dequantized_w = wrapper_layer.q_weight.float() * wrapper_layer.scale
    print("Dequantized weight:", dequantized_w)
    
    # 5) Forward pass with quantized model
    y_quant = quantized_model(x)
    print("Quantized model output:", y_quant)
    print("========================================================")


if __name__ == "__main__":
    test_linear()
    test_conv1d()
