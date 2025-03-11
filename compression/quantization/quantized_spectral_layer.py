import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorly as tl
from tensorly.tenalg import multi_mode_dot

from tltorch.factorized_tensors import FactorizedTensor

class QuantizedSpectralConv(nn.Module):
    """
    Demonstration of how to quantize a SpectralConv's factorized weights 
    (specifically Tucker factorization as an example) without always calling `to_tensor()`.
    
    Key Fixes:
    ----------
    1) We use register_buffer for int8 data - no nn.Parameter(int8).
    2) We attempt a minimal Tucker reconstruction in the forward pass 
       to show how factor-wise dequantization might happen.
    """

    def __init__(self, spectral_layer):
        super().__init__()

        # Store SpectralConv metadata
        self.in_channels = spectral_layer.in_channels
        self.out_channels = spectral_layer.out_channels
        self.n_modes = spectral_layer.n_modes
        self.order = len(self.n_modes)
        self.separable = spectral_layer.separable
        self.fft_norm = spectral_layer.fft_norm
        self.resolution_scaling_factor = spectral_layer.resolution_scaling_factor
        self.has_bias = spectral_layer.bias is not None

        # Extract the underlying weight object (could be factorized or dense)
        weight_obj = spectral_layer.weight  

        # Check for Tucker factorization
        if (isinstance(weight_obj, FactorizedTensor)
            and weight_obj.name.lower().endswith("tucker")):
            self.is_factorized = True

            # 1) Quantize the core if present
            if hasattr(weight_obj, "core"):
                q_core, scale_c, min_c = self._quantize_tensor(weight_obj.core)
                self.register_buffer("q_core", q_core)
                self.register_buffer("core_min", torch.tensor(min_c, dtype=torch.float32))
                self.core_scale = scale_c
            else:
                self.register_buffer("q_core", None)

            # 2) Quantize each factor separately
            self.factors = []           # list of buffer names
            self.factor_scales = []
            self.factor_mins = []

            for i, fmat in enumerate(weight_obj.factors):
                q_f, scale_f, min_f = self._quantize_tensor(fmat)
                buffer_name = f"factor_{i}"    # e.g. factor_0, factor_1, ...
                self.register_buffer(buffer_name, q_f)
                self.factors.append(buffer_name)

                self.factor_scales.append(scale_f)
                self.factor_mins.append(min_f)
        else:
            # Fallback: dense approach
            self.is_factorized = False
            # either call .to_tensor() or if it's already dense cfloat
            if hasattr(weight_obj, 'to_tensor'):
                dense_cplx = weight_obj.to_tensor()
            else:
                dense_cplx = weight_obj

            q_w, scale_w, min_w = self._quantize_tensor(dense_cplx)
            self.register_buffer('q_weight', q_w)
            self.register_buffer('w_min', torch.tensor(min_w, dtype=torch.float32))
            self.w_scale = scale_w

        # 3) Quantize the bias if present
        if self.has_bias:
            b = spectral_layer.bias
            q_b, b_scale, b_min = self._quantize_tensor(b)
            self.register_buffer('q_bias', q_b)
            self.register_buffer('b_min', torch.tensor(b_min, dtype=torch.float32))
            self.b_scale = b_scale
        else:
            self.register_buffer('q_bias', None)

    def _quantize_tensor(self, tensor: torch.Tensor):
        """
        Quantize a real or complex float tensor into int8 using min->max scaling.
        Returns: (q_tensor, scale, min_val)
        """
        print("---------------------------------------------------------------------------------")
        print("Tensor dtype is:", tensor.dtype)
        print("---------------------------------------------------------------------------------")
        if tensor.dtype in (torch.complex64, torch.complex128):
            # It's genuinely complex
            real_view = torch.view_as_real(tensor)  # shape [..., 2]
            min_val = real_view.min().item()
            max_val = real_view.max().item()
            scale = (max_val - min_val) / 256
            if scale == 0:
                scale = 1.0
            q_t = torch.round((real_view - min_val) / scale - 127).clamp(-127, 127).to(torch.int8)
            return q_t, scale, min_val
        else:
            # It's real float32 or float64
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            scale = (max_val - min_val) / 256
            if scale == 0:
                scale = 1.0
            q_t = torch.round((tensor - min_val) / scale - 127).clamp(-127, 127).to(torch.int8)
            return q_t, scale, min_val

    def _dequantize(self, q_int8: torch.Tensor, scale: float, min_val: float, is_complex: bool=False):
        """
        Helper to go from int8 -> float (optionally complex).
        If is_complex=True, the last dimension is 2, so we call torch.view_as_complex at the end.
        """
        f_val = (q_int8.float() + 127) * scale + min_val
        if is_complex:
            # shape [..., 2] => complex
            return torch.view_as_complex(f_val)
        return f_val

    def forward(self, x, output_shape=None):
        """
        1) rFFT over 2D freq domain
        2) If factorized => reconstruct float weight 
        else => dequantize dense weight
        3) For each frequency slice, accumulate over in_channels->out_channels
        4) iFFT
        5) add bias
        """

        import tensorly.tenalg as tltenalg  # if you need multi_mode_dot for factor code
        from tensorly.tenalg import multi_mode_dot

        # 1) rFFT
        # For 2D, typically self.order=2 => fft_dims = [-2, -1]
        fft_dims = list(range(-self.order, 0))
        x_fft = torch.fft.rfftn(x, dim=fft_dims, norm=self.fft_norm)  # shape ~ [B, Cin, Nx, Ny//2+1]

        # 2) reconstruct or dequantize => W_float with shape [Cin, Cout, NxW, NyW], or factor
        if self.is_factorized:
            # (same code as before) do Tucker reconstruction or fallback
            if hasattr(self, 'q_core') and self.q_core is not None:
                core_float = self._dequantize(
                    self.q_core, 
                    self.core_scale, 
                    self.core_min.item(),
                    is_complex=(self.q_core.ndim > 1 and self.q_core.shape[-1] == 2)
                )
            else:
                core_float = None

            deq_factors = []
            for i, f_name in enumerate(self.factors):
                q_f_int8 = getattr(self, f_name)
                scale_f = self.factor_scales[i]
                min_f = self.factor_mins[i]
                is_cplx = (q_f_int8.ndim > 1 and q_f_int8.shape[-1] == 2)
                f_float = self._dequantize(q_f_int8, scale_f, min_f, is_complex=is_cplx)
                deq_factors.append(f_float)

            if core_float is not None:
                modes = list(range(core_float.ndim))
                W_float = multi_mode_dot(core_float, deq_factors, modes=modes)
            else:
                W_float = deq_factors[0]
        else:
            # dense fallback
            q_w = self.q_weight
            min_w = self.w_min.item()
            scale_w = self.w_scale
            is_cplx = (q_w.ndim > 1 and q_w.shape[-1] == 2)
            W_float = self._dequantize(q_w, scale_w, min_w, is_complex=is_cplx)

        # shapes:
        # x_fft:   [B, Cin, Nx, Ny_r]
        # W_float: [Cin, Cout, NxW, NyW]  (assuming 2D PDE for example)

        B, Cin, Nx, Ny_r = x_fft.shape
        # Let's parse out the shape of the float weight:
        CinW, Cout, NxW, NyW = W_float.shape

        # Quick sanity check:
        if Cin != CinW:
            raise ValueError(f"Mismatch: x_fft has in_channels={Cin}, but W has {CinW}")

        # 3) We'll create out_fft with shape [B, Cout, Nx, Ny_r],
        #    then do partial indexing so we only multiply where NxW, NyW fits.
        out_fft = torch.zeros(
            (B, Cout, Nx, Ny_r),
            dtype=x_fft.dtype, device=x_fft.device
        )

        # Decide how many frequencies we keep in each dimension
        # This logic follows your partial slicing approach from spectral_convolution
        center_x = Nx // 2          # zero freq is near Nx//2
        neg_x = NxW // 2
        pos_x = NxW // 2 + (NxW % 2)
        slice_x = slice(center_x - neg_x, center_x + pos_x)  # range in x_fft dimension

        center_y = Ny_r // 2
        neg_y = NyW // 2
        pos_y = NyW // 2 + (NyW % 2)
        slice_y = slice(center_y - neg_y, center_y + pos_y)

        # (Optional) clamp the slices to avoid out-of-bounds if NxW> Nx, etc.
        # e.g.
        # slice_x = slice(max(0, center_x - neg_x), min(Nx, center_x + pos_x))
        # slice_y = ...
        # or replicate the "starts" logic from your code. This is just one approach.

        # Now we do double loop over Cin->Cout but only for the partial region
        # out_fft[:, c_out, slice_x, slice_y] += sum_{c_in}( x_fft[:, c_in, slice_x, slice_y] * W_float[c_in, c_out, :, :] )
        # Because NxW might not match the slice size exactly, we do something like:
        Wx_slice = slice(0, pos_x + neg_x)   # or NxW
        Wy_slice = slice(0, pos_y + neg_y)   # or NyW

        for c_in in range(Cin):
            for c_out in range(Cout):
                # Multiply for partial region
                out_fft[:, c_out, slice_x, slice_y] += (
                    x_fft[:, c_in, slice_x, slice_y]
                    * W_float[c_in, c_out, Wx_slice, Wy_slice]
                )

        # 4) iFFT
        x_out = torch.fft.irfftn(out_fft, s=x.shape[2:], dim=fft_dims, norm=self.fft_norm)

        # 5) Add bias if present
        if self.q_bias is not None:
            is_b_cplx = (self.q_bias.ndim > 1 and self.q_bias.shape[-1] == 2)
            b = self._dequantize(self.q_bias, self.b_scale, self.b_min.item(), is_complex=is_b_cplx)
            # broadcast across batch & freq dims
            x_out = x_out + b.view(1, -1, 1, 1)  # or [1, outC, Nx, Ny]

        return x_out



    def transform(self, x, output_shape=None):
        """
        This method is adapted from the original SpectralConv.transform, so that
        any domain resizing or resolution scaling still happens. If you want a true no-op,
        you'd just do 'return x'. But real FNO code often uses domain padding or
        resampling to match the number of modes.

        x : torch.Tensor of shape [batch_size, channels, spatial_dim1, spatial_dim2, ...]
        output_shape : optional. If provided, we resample x to that shape.
        """
        # in_shape is the current spatial shape after [B, C, ...]
        in_shape = list(x.shape[2:])

        # If you stored e.g. self.resolution_scaling_factor in __init__
        if self.resolution_scaling_factor is not None and output_shape is None:
            # scale each dimension in 'in_shape' by the corresponding factor
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            # user-specified shape
            out_shape = output_shape
        else:
            # keep the shape if nothing is specified
            out_shape = in_shape

        # If the shape is already correct, do nothing
        if in_shape == list(out_shape):
            return x
        else:
            # Resample is presumably the same function from your original code
            # that changes the spatial resolution of x
            from .resample import resample  # or wherever `resample` is imported
            return resample(x, scale_factor=1.0, dims=list(range(2, x.ndim)), output_shape=out_shape)

