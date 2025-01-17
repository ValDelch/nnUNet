from typing import Tuple, List, Union, Type

import numpy as np
import torch
from torch import nn as torch_nn
from escnn import nn as e2_nn
from escnn.group import directsum

from dynamic_network_architectures.building_blocks.helper_e2cnn import maybe_convert_scalar_to_list, convert_conv_op_to_dim


class ConvertToTensor(e2_nn.EquivariantModule):
    def __init__(self,
                 in_type: Type[e2_nn.FieldType]):
        super(ConvertToTensor, self).__init__()
        self.out_type = in_type
        if self.out_type.gspace.dimensionality == 2:
            self.gpool = e2_nn.GroupPooling(in_type)
        #else:
        #    self.gs = self.out_type.gspace
        #    ftgpool = e2_nn.QuotientFourierELU(self.gs, (False, -1), 128, 
        #                                       irreps=self.gs.fibergroup.bl_irreps(2), 
        #                                       out_irreps=self.gs.fibergroup.bl_irreps(0), 
        #                                       grid=self.gs.fibergroup.sphere_grid(type='thomson_cube', N=1))
        #    final_features = ftgpool.in_type
        #    conv = e2_nn.R3Conv(self.out_type, final_features, kernel_size=3, padding=0, bias=False, initialize=False)
        #    self.gpool = e2_nn.SequentialModule(conv, ftgpool)
        else:
            self.gpool = e2_nn.NormPool(in_type)

    def forward(self, x):
        x = self.gpool(x)
        return x.tensor
    
    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        pass


class ConvDropoutNormReLU(e2_nn.EquivariantModule):
    def __init__(self,
                 conv_op: Type[e2_nn.EquivariantModule],
                 in_type: Type[e2_nn.FieldType],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[e2_nn.EquivariantModule]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[e2_nn.EquivariantModule]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[e2_nn.EquivariantModule]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(ConvDropoutNormReLU, self).__init__()

        self.in_type = in_type
        self.gspace = in_type.gspace
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, initial_stride)
        self.stride = stride
        self.dim = convert_conv_op_to_dim(conv_op)

        if (type(kernel_size) is tuple) or (type(kernel_size) is list):
            kernel_size = kernel_size[0]
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        ops = []

        if self.gspace.dimensionality == 2:

            # Adding the e2 convolution layer
            out_type = e2_nn.FieldType(in_type.gspace, output_channels*[in_type.gspace.regular_repr])
            self.out_type = out_type
            self.conv = conv_op(
                in_type=in_type, 
                out_type=out_type,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                dilation=1,
                bias=conv_bias,
            )
            ops.append(self.conv)

            if dropout_op is not None:
                self.dropout = dropout_op(out_type, **dropout_op_kwargs)
                ops.append(self.dropout)

            if norm_op is not None:
                self.norm = norm_op(out_type, **norm_op_kwargs)
                ops.append(self.norm)

            if nonlin is not None:
                self.nonlin = nonlin(out_type, **nonlin_kwargs)
                ops.append(self.nonlin)

            if nonlin_first and (norm_op is not None and nonlin is not None):
                ops[-1], ops[-2] = ops[-2], ops[-1]

        else:

            SO3 = self.gspace.fibergroup
            polinomials = [self.gspace.trivial_repr, SO3.irrep(1)]
            for _ in range(2, 3):
                polinomials.append(
                    polinomials[-1].tensor(SO3.irrep(1))
                )
            out_type = directsum(polinomials, name=f'polynomial_2')
            out_type = e2_nn.FieldType(self.gspace, [out_type] * output_channels)
            self.out_type = out_type

            ops.append(conv_op(
                in_type=self.in_type,
                out_type=out_type,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                dilation=1,
                bias=conv_bias
            ))

            if norm_op is not None:
                self.norm = e2_nn.IIDBatchNorm3d(out_type, **norm_op_kwargs)
                ops.append(self.norm)

            if nonlin is not None:
                self.nonlin = e2_nn.NormNonLinearity(out_type)
                ops.append(self.nonlin)

        self.all_modules = e2_nn.SequentialModule(*ops)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = e2_nn.GeometricTensor(x, self.in_type)
        return self.all_modules(x)
    
    def compute_conv_feature_map_size(self, input_size, order=4):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels*(order**(self.dim-1)), *output_size], dtype=np.int64)
    
    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        pass


class StackedConvBlocks(e2_nn.EquivariantModule):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[e2_nn.EquivariantModule],
                 in_type: Type[e2_nn.FieldType],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[e2_nn.EquivariantModule]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[e2_nn.EquivariantModule]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[e2_nn.EquivariantModule]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super().__init__()

        if not isinstance(output_channels, (list, tuple)):
            output_channels = num_convs * [output_channels]

        gspace = in_type.gspace
        self.gspace = gspace

        convs = []
        for i in range(num_convs):
            if i == 0:
                convs.append(
                    ConvDropoutNormReLU(
                        conv_op=conv_op,
                        in_type=in_type,
                        input_channels=input_channels,
                        output_channels=output_channels[0],
                        kernel_size=kernel_size,
                        initial_stride=initial_stride,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        dropout_op=dropout_op,
                        dropout_op_kwargs=dropout_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        nonlin_first=nonlin_first
                    )
                )
            else:
                convs.append(
                    ConvDropoutNormReLU(
                        conv_op=conv_op,
                        in_type=convs[i-1].out_type,
                        input_channels=output_channels[i-1],
                        output_channels=output_channels[i],
                        kernel_size=kernel_size,
                        initial_stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        dropout_op=dropout_op,
                        dropout_op_kwargs=dropout_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        nonlin_first=nonlin_first
                    )
                )
        self.convs = torch_nn.Sequential(
            *convs
        )

        self.output_channels = output_channels[-1]
        self.out_type = self.convs[-1].out_type
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)
    
    def compute_conv_feature_map_size(self, input_size, order):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size, order)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]  # we always do same padding
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride, order)
        return output
    
    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        pass
    

if __name__ == '__main__':

    from escnn import gspaces

    # Test 2D
    input_channels = 3
    data = torch.rand(1, input_channels, 40, 32)

    r2_act = gspaces.rot2dOnR2(N=4)
    in_type = e2_nn.FieldType(r2_act, input_channels*[r2_act.trivial_repr])

    stx = StackedConvBlocks(
        num_convs=2,
        conv_op=e2_nn.R2Conv,
        in_type=in_type,
        input_channels=input_channels,
        output_channels=16,
        kernel_size=5,
        initial_stride=2,
        conv_bias=True,
        norm_op=e2_nn.InnerBatchNorm,
        norm_op_kwargs=None,
        dropout_op=e2_nn.PointwiseDropout,
        dropout_op_kwargs={'p': 0.1},
        nonlin=e2_nn.ELU,
        nonlin_kwargs={'alpha': 0.1, 'inplace': True}
    )

    model = torch_nn.Sequential(
        stx,
        ConvDropoutNormReLU(
            conv_op=e2_nn.R2Conv,
            in_type=stx.convs[-1].out_type,
            input_channels=stx.output_channels,
            output_channels=32,
            kernel_size=(7,7),
            initial_stride=1,
            conv_bias=True,
            norm_op=e2_nn.InnerBatchNorm,
            norm_op_kwargs=None,
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=e2_nn.ReLU,
            nonlin_kwargs={'inplace': True}
        )
    )

    model = torch_nn.Sequential(
        model,
        ConvertToTensor(in_type=model[-1].out_type)
    )

    print(model(data).shape)
    print(type(model(data)))

    print(stx.compute_conv_feature_map_size((40, 32), 4))

    # Test 3D
    input_channels = 3
    data = torch.rand(1, input_channels, 40, 32, 32)

    r2_act = gspaces.rot3dOnR3()
    in_type = e2_nn.FieldType(r2_act, input_channels*[r2_act.trivial_repr])

    stx = StackedConvBlocks(
        num_convs=2,
        conv_op=e2_nn.R3Conv,
        in_type=in_type,
        input_channels=input_channels,
        output_channels=4,
        kernel_size=5,
        initial_stride=2,
        conv_bias=True,
        norm_op=e2_nn.InnerBatchNorm,
        norm_op_kwargs=None,
        dropout_op=e2_nn.PointwiseDropout,
        dropout_op_kwargs={'p': 0.1},
        nonlin=e2_nn.ELU,
        nonlin_kwargs={'alpha': 0.1, 'inplace': True}
    )

    model = torch_nn.Sequential(
        stx,
        ConvDropoutNormReLU(
            conv_op=e2_nn.R3Conv,
            in_type=stx.convs[-1].out_type,
            input_channels=stx.output_channels,
            output_channels=16,
            kernel_size=(7,7,7),
            initial_stride=1,
            conv_bias=True,
            norm_op=e2_nn.InnerBatchNorm,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=e2_nn.ReLU,
            nonlin_kwargs={'inplace': True}
        )
    )

    model = torch_nn.Sequential(
        model,
        ConvertToTensor(in_type=model[-1].out_type)
    )

    print(model(data).shape)
    print(type(model(data)))

    print(stx.compute_conv_feature_map_size((40, 32, 32), 4))