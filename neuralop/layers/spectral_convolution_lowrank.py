from typing import List, Optional, Tuple, Union

from ..utils import validate_scaling_factor

import torch
from torch import nn

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor


from typing import List, Optional, Tuple, Union
from copy import deepcopy
from ..utils import validate_scaling_factor
import torch
from torch import nn
import tensorly as tl
from tltorch.factorized_tensors.core import FactorizedTensor
from .base_spectral_conv import BaseSpectralConv

from .einsum_utils import einsum_complexhalf
from .base_spectral_conv import BaseSpectralConv
from .resample import resample

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    
    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    if x.dtype == torch.complex32:
        # if x is half precision, run a specialized einsum
        return einsum_complexhalf(eq, x, weight)
    else:
        return tl.einsum(eq, x, weight)

def _contract_dense_separable(x, weight, separable):
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    return x * weight

def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...
    eq = f'{x_syms},{rank_sym},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, cp_weight.weights, *cp_weight.factors)
    else:
        return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        # x, y, ...
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]

    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        # x, y, ...
        factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, tucker_weight.core, *tucker_weight.factors)
    else:
        return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, *tt_weight.factors)
    else:
        return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)
    separable: bool
        if True, performs contraction with individual tensor factors. 
        if False, 
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("tt"):
                return _contract_tt
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got implementation={implementation}, expected "reconstructed" or "factorized"'
        )


Number = Union[int, float]

# class DoubleSpectralConv(torch.nn.Module):
#     """正确的双权重谱卷积层，仅在频域执行两次矩阵乘法"""
#     def __init__(self, 
#                  in_channels: int,
#                  out_channels: int,
#                  n_modes: Tuple[int],
#                  mid_channels: int, # 中间层的通道数（即SVD的rank）
#                  **kwargs):
#         super().__init__()
        
#         # 初始化两个分解后的权重矩阵
#         self.weight1 = torch.nn.Parameter( # [in_channels, mid_channels, modes_x, modes_y]
#             torch.randn(in_channels, mid_channels, *n_modes, dtype=torch.cfloat))
#         self.weight2 = torch.nn.Parameter( # [mid_channels, out_channels, modes_x, modes_y]
#             torch.randn(mid_channels, out_channels, *n_modes, dtype=torch.cfloat))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """ 
#         修正后的前向传播：
#         1. 只做一次FFT
#         2. 连续应用两个权重
#         3. 只做一次IFFT
#         """
#         # Step 1: 转换到频域 (仅一次FFT)
#         x_ft = torch.fft.rfft2(x, norm="ortho") # [B, C_in, H, W//2+1]
        
#         # Step 2: 在频域连续应用两个权重
#         # 第一次矩阵乘法: [B, C_in, H, W] x [C_in, mid] -> [B, mid, H, W]
#         x_ft = torch.einsum('bihw,iohw->bohw', x_ft, self.weight1)
        
#         # 第二次矩阵乘法: [B, mid, H, W] x [mid, out] -> [B, out, H, W]
#         x_ft = torch.einsum('bihw,iohw->bohw', x_ft, self.weight2)
        
#         # Step 3: 转换回空间域 (仅一次IFFT)
#         x = torch.fft.irfft2(x_ft, s=x.shape[-2:], norm="ortho") # [B, C_out, H, W]
#         return x

class DoubleSpectralConv(BaseSpectralConv):
    """双权重谱卷积层，在频域依次应用两个独立权重"""
    
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
        mid_channels=None,  # 新增中间通道数参数
    ):
        super().__init__(device=device)
        
        # 参数校验与初始化
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels or out_channels  # 默认中间通道等于输出通道
        self.complex_data = complex_data
        self.n_modes = n_modes
        self.order = len(self.n_modes)
        self.max_n_modes = max_n_modes or self.n_modes
        self.fno_block_precision = fno_block_precision
        self.separable = separable
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation
        self.resolution_scaling_factor = validate_scaling_factor(resolution_scaling_factor, self.order)
        self.fft_norm = fft_norm

        # 权重初始化逻辑
        if init_std == "auto":
            init_std1 = (2 / (in_channels + self.mid_channels))**0.5
            init_std2 = (2 / (self.mid_channels + out_channels))**0.5
        else:
            init_std1 = init_std2 = init_std

        # 创建第一个权重
        self.weight1 = self._create_weight(
            in_channels, self.mid_channels, 
            init_std1, decomposition_kwargs or {}
        )
        
        # 创建第二个权重 
        self.weight2 = self._create_weight(
            self.mid_channels, out_channels,
            init_std2, deepcopy(decomposition_kwargs) or {}
        )

        # 偏置项
        if bias:
            self.bias = nn.Parameter(
                init_std2 * torch.randn(*(tuple([out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

    def _create_weight(self, in_ch, out_ch, init_std, decomp_kwargs):
        """权重创建辅助函数"""
        # 确定权重形状
        if self.separable:
            if in_ch != out_ch:
                raise ValueError("可分离卷积要求输入输出通道相同")
            weight_shape = (in_ch, *self.max_n_modes)
        else:
            weight_shape = (in_ch, out_ch, *self.max_n_modes)

        # 创建因子化或密集权重
        if self.factorization is None:
            weight = torch.empty(weight_shape, dtype=torch.cfloat, device=self.device)
        else:
            weight = FactorizedTensor.new(
                weight_shape, rank=self.rank,
                factorization=self.factorization,
                fixed_rank_modes=fixed_rank_modes,
                **decomp_kwargs
            )
        
        # 初始化权重
        weight.normal_(0, init_std)
        return weight

    def _get_contract_fn(self, weight):
        """获取对应权重的contract函数"""
        if self.implementation == "reconstructed":
            return _contract_dense_separable if self.separable else _contract_dense
        elif self.implementation == "factorized":
            if isinstance(weight, FactorizedTensor):
                if 'tucker' in weight.name.lower():
                    return partial(_contract_tucker, separable=self.separable)
                elif 'cp' in weight.name.lower():
                    return partial(_contract_cp, separable=self.separable)
                elif 'tt' in weight.name.lower():
                    return _contract_tt
            return _contract_dense
    
    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """修改后的前向传播，依次应用两个权重"""
        
        # 原始FFT变换逻辑保持不变
        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-self.order, 0))

        # 转换到频域
        if self.complex_data:
            x_ft = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
        else:
            x_ft = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)

        if self.order > 1:
            x_ft = torch.fft.fftshift(x_ft, dim=fft_dims[:-1])

        # 第一次权重应用
        intermediate = self._apply_weight(x_ft, self.weight1, fft_size)
        
        # 第二次权重应用
        out_ft = self._apply_weight(intermediate, self.weight2, fft_size)

        # 逆变换回空间域
        if self.order > 1:
            out_ft = torch.fft.ifftshift(out_ft, dim=fft_dims[:-1])

        if self.complex_data:
            x = torch.fft.ifftn(out_ft, s=output_shape, dim=fft_dims, norm=self.fft_norm)
        else:
            x = torch.fft.irfftn(out_ft, s=output_shape, dim=fft_dims, norm=self.fft_norm)

        if self.bias is not None:
            x += self.bias

        return x

    def _apply_weight(self, x_ft, weight, fft_size):
        """权重应用核心逻辑"""
        # 获取模态切片
        slices_w, slices_x = self._get_slices(fft_size, weight.shape)
        
        # 执行contract操作
        contract_fn = self._get_contract_fn(weight)
        out_ft = torch.zeros_like(x_ft)
        out_ft[slices_x] = contract_fn(x_ft[slices_x], weight[slices_w])
        
        return out_ft

    def _get_slices(self, fft_size, weight_shape):
        """生成权重和输入的切片索引"""
        # 权重切片逻辑（与原实现相同）
        starts = [max_size - min(size, n_mode) 
                for max_size, size, n_mode in zip(self.max_n_modes, fft_size, self.n_modes)]
        
        if self.separable:
            slices_w = [slice(None)]
        else:
            slices_w = [slice(None), slice(None)]
        
        slices_w += self._get_spatial_slices(starts)

        # 输入切片逻辑（与原实现相同）
        slices_x = [slice(None), slice(None)]
        for size, n_mode in zip(fft_size, self.n_modes):
            center = size // 2
            neg = n_mode // 2
            pos = n_mode // 2 + n_mode % 2
            slices_x.append(slice(center-neg, center+pos))
        
        return slices_w, slices_x

    def _get_spatial_slices(self, starts):
        """生成空间维度切片"""
        slices = []
        for i, start in enumerate(starts):
            if i == len(starts)-1 and not self.complex_data:
                slices.append(slice(None, -start) if start else slice(None))
            else:
                slices.append(slice(start//2, -start//2) if start else slice(None))
        return slices

