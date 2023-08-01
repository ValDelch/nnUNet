from typing import Tuple, List, Union, Type
from xmlrpc.client import Boolean
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.voxel_convolution import Convolution
from e3nn.nn import BatchNorm, Gate, Dropout
from e3nn.o3 import Irreps, Linear


class ConvDropoutNormReLU(nn.Module):
    def __init__(self,
                 equivariance: str,
                 num_radial_basis: int,
                 steps: tuple,
                 input_irreps: str,
                 n: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 cutoff: bool = True,
                 conv_bias: bool = True,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Module]] = None,
                 dropout_prob: float = 0.1,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 lmax: int = 2
                 ):
        super(ConvDropoutNormReLU, self).__init__()
        
        if isinstance(stride, List) or isinstance(stride, Tuple):
            stride = stride[0]
        self.stride = stride
        if isinstance(kernel_size, List) or isinstance(kernel_size, Tuple):
            kernel_size = kernel_size[0]

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        assert equivariance in ['SO3', 'O3']

        if equivariance == 'SO3':
            activation = [torch.relu]
            irreps_sh = Irreps.spherical_harmonics(lmax, 1)
            ne = n
            no = 0
        elif equivariance == 'O3':
            activation = [torch.relu,torch.tanh]
            irreps_sh = Irreps.spherical_harmonics(lmax, -1)
            ne = n
            no = n

        self.output_size = 7*ne + 7*no
        self.ne = ne ; self.no = no

        irreps_hidden = Irreps(f"{4*ne}x0e + {4*no}x0o + {2*ne}x1e + {2*no}x1o + {ne}x2e + {no}x2o").simplify()
        irreps_scalars = Irreps( [ (mul, ir) for mul, ir in irreps_hidden if ir.l == 0 ] )
        irreps_gated = Irreps( [ (mul, ir) for mul, ir in irreps_hidden if ir.l > 0 ] )
        irreps_gates = Irreps(f"{irreps_gated.num_irreps}x0e")
        if irreps_gates.dim == 0:
            irreps_gates = irreps_gates.simplify()
            activation_gate = []
        else:
            activation_gate = [torch.sigmoid]

        ops = []

        self.gate = Gate(irreps_scalars, activation, irreps_gates, activation_gate, irreps_gated)
        self.conv = Convolution(input_irreps, self.gate.irreps_in, irreps_sh, kernel_size, num_radial_basis, steps, cutoff)

        ops.append(self.gate)
        ops.append(self.conv)

        self.norm_op = norm_op
        if norm_op is not None:
            self.batchnorm = partial(BatchNorm, instance=True)(self.gate.irreps_in)
            ops.append(self.batchnorm)
        
        self.dropout_op = dropout_op
        if dropout_op is not None:
            self.dropout = Dropout(self.gate.irreps_out, dropout_prob)
            ops.append(self.dropout)

    def forward(self, x):
        
        x = self.conv(x)
        if self.norm_op is not None:
            x = self.batchnorm(x.transpose(1, 4)).transpose(1, 4)
        x = self.gate(x.transpose(1, 4)).transpose(1, 4)
        if self.dropout_op is not None:
            x = self.dropout(x.transpose(1, 4)).transpose(1, 4)
        if self.stride != 1:
            x = F.avg_pool3d(x, self.stride, stride=self.stride)
        return x

    def compute_conv_feature_map_size(self, input_size):
        stride = (self.stride,)*3
        assert len(input_size) == len(stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, stride)]  # we always do same padding
        return np.prod([self.output_size, *output_size], dtype=np.int64)


class StackedConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 equivariance: str,
                 num_radial_basis: int,
                 steps: tuple,
                 input_irreps: str,
                 n: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 cutoff: bool = True,
                 conv_bias: bool = True,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Module]] = None,
                 dropout_prob: float = 0.1,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False,
                 lmax: int = 2
                 ):
        """

        :param conv_op:
        :param num_convs:
        :param input_channels:
        :param output_channels: can be int or a list/tuple of int. If list/tuple are provided, each entry is for
        one conv. The length of the list/tuple must then naturally be num_convs
        :param kernel_size:
        :param initial_stride:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        """
        super().__init__()
        if not isinstance(n, (tuple, list)):
            n = [n] * num_convs

        if isinstance(initial_stride, List) or isinstance(initial_stride, Tuple):
            initial_stride = initial_stride[0]
        self.initial_stride = initial_stride

        self.convs = [ConvDropoutNormReLU(
                equivariance, num_radial_basis, steps, input_irreps, n[0], kernel_size,
                initial_stride, True, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_prob, nonlin, nonlin_kwargs, nonlin_first
            )
        ]
        for i in range(1, num_convs):
            self.convs.append(
                ConvDropoutNormReLU(
                    equivariance, num_radial_basis, steps, self.convs[-1].gate.irreps_out, n[i], kernel_size,
                    1, True, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_prob, nonlin, nonlin_kwargs, nonlin_first
                )
            )
        self.irreps_out = self.convs[-1].gate.irreps_out
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        initial_stride = (self.initial_stride,) * 3
        assert len(input_size) == len(initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


if __name__ == '__main__':
    data = torch.rand((1, 3, 40, 32, 32))

    stx = StackedConvBlocks(
        2,
        "SO3",
        5,
        (1,1,1),
        "3x0e",
        2,
        5,
        2,
        True
    )

    model = nn.Sequential(
        stx,
        ConvDropoutNormReLU(
            "SO3",
            5,
            (1,1,1),
            stx.convs[-1].gate.irreps_out,
            2,
            5,
            2,
            True
        )
    )

    print(model(data).shape)
    print(model[0].compute_conv_feature_map_size((40, 32, 32)))