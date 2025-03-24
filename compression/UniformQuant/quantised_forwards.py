import torch
from neuralop.models.fno import FNO
from neuralop.layers.fno_block import *
from neuralop.layers.skip_connections import *

def true_quantize(x : torch.Tensor):
    if not x.is_quantized:
        return torch.quantize_per_tensor_dynamic(x, torch.quint8, False)
    return x

def true_dequantize(x : torch.Tensor):
    if x.is_quantized:
        return x.dequantize()
    return x

def quantised_gating_forward(self : SoftGating, x):
        """Applies soft-gating to a batch of activations"""
        if self.bias is not None:
            return self.weight * true_dequantize(x) + self.bias
        else:
            return self.weight * true_dequantize(x)

def quantised_forward_with_postactivation(self : FNOBlocks, x: torch.Tensor, index=0, output_shape=None):
        x_skip_fno = self.fno_skips[index](true_quantize(x))
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](x)
        x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)
        #self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](true_quantize(x_fno))

        x = true_quantize(true_dequantize(x_fno) + true_dequantize(x_skip_fno))

        if (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        x = true_quantize(true_dequantize(self.channel_mlp[index](x)) + x_skip_channel_mlp)

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x

def quantised_forward_with_preactivation(self, x, index=0, output_shape=None):
    # Apply non-linear activation (and norm)
    # before this block's convolution/forward pass:
    x = self.non_linearity(x)

    if self.norm is not None:
        x = self.norm[self.n_norms * index](x)

    x_skip_fno = self.fno_skips[index](x)
    x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

    x_skip_channel_mlp = self.channel_mlp_skips[index](x)
    x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

    if self.stabilizer == "tanh":
        if self.complex_data:
            x = ctanh(x)
        else:
            x = torch.tanh(x)

    x_fno = self.convs[index](x, output_shape=output_shape)

    x = x_fno + x_skip_fno

    if index < (self.n_layers - 1):
        x = self.non_linearity(x)

    if self.norm is not None:
        x = self.norm[self.n_norms * index + 1](x)

    x = self.channel_mlp[index](x) + x_skip_channel_mlp

    return x

def quantised_fno_forward(self : FNO, x: torch.Tensor, output_shape=None, **kwargs):
        """FNO's forward pass
        
        1. Applies optional positional encoding

        2. Sends inputs through a lifting layer to a high-dimensional latent space

        3. Applies optional domain padding to high-dimensional intermediate function representation

        4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity) 

        5. If domain padding was applied, domain padding is removed

        6. Projection of intermediate function representation to the output channels

        Parameters
        ----------
        x : tensor
            input tensor
        
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            
            * If None, don't specify an output shape

            * If tuple, specifies the output-shape of the **last** FNO Block

            * If tuple list, specifies the exact output-shape of each FNO Block
        """

        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]
        
        x = true_quantize(x)
        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = true_dequantize(x)
            x = self.positional_embedding(x)
            x = true_quantize(x)
        
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = true_dequantize(x)
            x = self.domain_padding.pad(x)
            x = true_quantize(x)

        for layer_idx in range(self.n_layers):
            x = true_dequantize(x)
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])
            x = true_quantize(x)

        if self.domain_padding is not None:
            x = true_dequantize(x)
            x = self.domain_padding.unpad(x)
            x = true_quantize(x)

        x = self.projection(x)

        return true_dequantize(x)