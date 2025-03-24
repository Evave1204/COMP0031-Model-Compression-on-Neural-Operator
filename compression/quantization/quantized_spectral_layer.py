'''
after trail and try we found that the int8 can decrease model size to 1/4 but with error increase from 13.26% to 13.29%
so we decide to use int8 finally
if want to change to int16 justneed to use 65535 to replace 256 and 32767 replace 127
also 
q_val = torch.round((val - val_min) / scale - 127).clamp(-127, 127).to(torch.int8)
has to change to 
q_val = torch.round((val - val_min) / scale - 32768).clamp(-32768, 32767).to(torch.int16)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizedSpectralConv(nn.Module):
    """
    A 'quantized' version of SpectralConv using int16.
    

    Changes from the int8 version:
      - Uses int16 with a range of -32768 to 32767.
      - Scale is computed as (max - min) / 65535.0.
      - Offset is 32768.
      - For bias quantization, we now **do not clamp** the scale (or you could set a much lower clamp) so that the very small bias range is preserved.
    """

    def __init__(self, spectral_layer):
        super().__init__()

        # ---- 1) Copy metadata from the original layer ----
        self.in_channels  = spectral_layer.in_channels
        self.out_channels = spectral_layer.out_channels
        self.n_modes      = spectral_layer.n_modes  # e.g. (12, 12)
        self.order        = len(self.n_modes)
        self.separable    = spectral_layer.separable
        self.fft_norm     = spectral_layer.fft_norm
        self.resolution_scaling_factor = spectral_layer.resolution_scaling_factor

        self.has_bias = spectral_layer.bias is not None

        if hasattr(spectral_layer, 'complex_data'):
            self.complex_data = spectral_layer.complex_data
        else:
            self.complex_data = False

        # ---- 2) Extract the original (dense) spectral weight ----
        weight_obj = spectral_layer.weight
        if hasattr(weight_obj, 'to_tensor'):
            weight = weight_obj.to_tensor()
        else:
            weight = weight_obj

        # ---- 3) Quantize the weight (handle complex vs. real) using int16 ----
        if weight.is_complex():
            (q_real, sr, mr, q_imag, si, mi) = self._quantize_tensor_int16(weight)
            self.register_buffer('q_real', q_real)
            self.register_buffer('q_imag', q_imag)
            self.register_buffer('min_real', torch.tensor(mr, dtype=torch.float32))
            self.register_buffer('min_imag', torch.tensor(mi, dtype=torch.float32))
            self.scale_real = sr
            self.scale_imag = si
            self.register_buffer('q_weight', None)
        else:
            (q_w, scale_w, min_w, _, _, _) = self._quantize_tensor_int16(weight)
            self.register_buffer('q_weight', q_w)
            self.register_buffer('w_min', torch.tensor(min_w, dtype=torch.float32))
            self.w_scale = scale_w
            self.register_buffer('q_real', None)
            self.register_buffer('q_imag', None)

        # ---- 4) Quantize bias if present (using int16) ----
        if self.has_bias:
            bias = spectral_layer.bias
            (q_b, b_scale, b_min, _, _, _) = self._quantize_tensor_int16(bias)
            # For debugging, we remove or lower the bias scale clamp so that the computed small scale is used.
            # b_scale = max(1e-3, b_scale)  <-- REMOVED
            self.register_buffer('q_bias', q_b)
            self.register_buffer('b_min', torch.tensor(b_min, dtype=torch.float32))
            self.b_scale = b_scale
        else:
            self.register_buffer('q_bias', None)

    # ------------------------------------------------------------------
    #  Quantization helper methods for int16
    # ------------------------------------------------------------------

    def _quantize_tensor_int16(self, tensor: torch.Tensor):
        """
        Quantize a tensor to int16 using min->max scaling.
        For int16, we use a range of 65535 steps and an offset of 32768.
        
        Returns:
          If tensor is complex:
            (q_real, scale_real, min_real, q_imag, scale_imag, min_imag)
          If real:
            (q_tensor, scale, min_val, None, None, None)
        """
        # cahnegs strat from here
        def quantize_real(val: torch.Tensor):
            val_min = val.min().item()
            val_max = val.max().item()
            scale = (val_max - val_min) / 256.0
            if scale == 0:
                scale = 1.0
            q_val = torch.round((val - val_min) / scale - 127).clamp(-127, 128).to(torch.int8)
            return q_val, scale, val_min

        if tensor.is_complex():
            real_part = tensor.real
            imag_part = tensor.imag

            q_real, r_scale, r_min = quantize_real(real_part)
            q_imag, i_scale, i_min = quantize_real(imag_part)
            return (q_real, r_scale, r_min, q_imag, i_scale, i_min)
        else:
            q_t, scale, val_min = quantize_real(tensor)
            return (q_t, scale, val_min, None, None, None)

    def _dequantize_int16(self, q_int16: torch.Tensor, scale: float, min_val: float) -> torch.Tensor:
        """
        Dequantize an int16 tensor back to float.
        """
        return (q_int16.float() + 127) * scale + min_val

    def _dequantize_weight(self) -> torch.Tensor:
        """
        Dequantize the stored spectral weight.
        """
        if self.q_real is not None and self.q_imag is not None:
            W_real = self._dequantize_int16(self.q_real, self.scale_real, self.min_real.item())
            W_imag = self._dequantize_int16(self.q_imag, self.scale_imag, self.min_imag.item())
            return torch.complex(W_real, W_imag)
        elif self.q_weight is not None:
            return self._dequantize_int16(self.q_weight, self.w_scale, self.w_min.item())
        else:
            raise RuntimeError("No stored weight buffers found for the spectral weight.")

    # ------------------------------------------------------------------
    #  Forward method using your old slicing approach
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, output_shape=None) -> torch.Tensor:
        """
        Forward pass:
         1) FFT (rFFTN or FFTN)
         2) Dequantize weight
         3) Center-slice frequency domain & perform multiplication
         4) iFFT back to spatial domain
         5) Add bias (if present)
        """
        fft_dims = list(range(-self.order, 0))
        if not self.complex_data:
            x_fft = torch.fft.rfftn(x, dim=fft_dims, norm=self.fft_norm)
        else:
            x_fft = torch.fft.fftn(x, dim=fft_dims, norm=self.fft_norm)
        B, Cin, Nx, Ny_r = x_fft.shape

        W_float = self._dequantize_weight()
        CinW, CoutW, NxW, NyW = W_float.shape
        if CinW != Cin:
            raise ValueError(f"Mismatch in in_channels: x_fft has {Cin}, weight has {CinW}")

        out_fft = torch.zeros((B, CoutW, Nx, Ny_r), dtype=x_fft.dtype, device=x_fft.device)

        center_x = Nx // 2
        neg_x = NxW // 2
        pos_x = NxW - neg_x
        slice_x = slice(center_x - neg_x, center_x + pos_x)

        center_y = Ny_r // 2
        neg_y = NyW // 2
        pos_y = NyW - neg_y
        slice_y = slice(center_y - neg_y, center_y + pos_y)

        Wx_slice = slice(0, NxW)
        Wy_slice = slice(0, NyW)

        for c_in in range(Cin):
            for c_out in range(CoutW):
                out_fft[:, c_out, slice_x, slice_y] += (
                    x_fft[:, c_in, slice_x, slice_y] *
                    W_float[c_in, c_out, Wx_slice, Wy_slice]
                )

        if not self.complex_data:
            x_out = torch.fft.irfftn(out_fft, s=x.shape[2:], dim=fft_dims, norm=self.fft_norm)
        else:
            x_out = torch.fft.ifftn(out_fft, s=x.shape[2:], dim=fft_dims, norm=self.fft_norm)

        if self.has_bias and self.q_bias is not None:
            b_float = self._dequantize_int16(self.q_bias, self.b_scale, self.b_min.item())
            x_out = x_out + b_float.view(1, -1, 1, 1)

        return x_out

    def transform(self, x, output_shape=None):
        return x
