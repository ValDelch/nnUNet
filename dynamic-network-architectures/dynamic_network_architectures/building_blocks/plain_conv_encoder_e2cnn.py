from typing import Tuple, List, Union, Type

import numpy as np
import torch
from torch import nn as torch_nn
from e2cnn import nn as e2_nn

from dynamic_network_architectures.building_blocks.simple_conv_blocks_e2cnn import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper_e2cnn import maybe_convert_scalar_to_list, get_matching_pool_op


class PlainConvEncoder(e2_nn.EquivariantModule):
    def __init__(self,
                 gspace: Type[e2_nn.FieldType],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[e2_nn.EquivariantModule],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False, 
                 norm_op: Union[None, Type[e2_nn.EquivariantModule]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[e2_nn.EquivariantModule]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[e2_nn.EquivariantModule]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv'
                 ):
        super().__init__()

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, f"kernel_sizes must be of length {n_stages}"
        assert len(n_conv_per_stage) == n_stages, f"n_conv_per_stage must be of length {n_stages}"
        assert len(features_per_stage) == n_stages, f"features_per_stage must be of length {n_stages}"
        assert len(strides) == n_stages, f"strides must be of length {n_stages}"

        in_type = e2_nn.FieldType(gspace, input_channels*[gspace.trivial_repr])
        self.in_type = in_type

        stages = []
        self.out_types = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        (isinstance(strides[s], (tuple, list))) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(
                        get_matching_pool_op(conv_op, pool_type=pool)(in_type=in_type, kernel_size=strides[s], stride=strides[s])
                    )
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(
                StackedConvBlocks(
                    n_conv_per_stage[s], conv_op, in_type, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride, 
                    conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
            )
            stages.append(torch_nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]
            in_type = stage_modules[-1].out_type
            self.out_types.append(in_type)

        self.stages = torch_nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.gspace = gspace
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = e2_nn.GeometricTensor(x, self.in_type)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]
        
    def compute_conv_feature_map_size(self, input_size, order=4):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], torch_nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size, order)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size, order)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output
    
    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        pass


if __name__ == '__main__':

    from e2cnn import gspaces

    # Test
    input_channels = 3
    data = torch.rand(1, input_channels, 64, 128)

    r2_act = gspaces.Rot2dOnR2(N=4)
    in_type = e2_nn.FieldType(r2_act, input_channels*[r2_act.trivial_repr])

    model = PlainConvEncoder(
        gspace=r2_act,
        input_channels=input_channels,
        n_stages=4,
        features_per_stage=16,
        conv_op=e2_nn.R2Conv,
        kernel_sizes=5,
        strides=2,
        n_conv_per_stage=2,
        conv_bias=True,
        norm_op=e2_nn.InnerBatchNorm,
        norm_op_kwargs=None,
        dropout_op=e2_nn.PointwiseDropout,
        dropout_op_kwargs={'p': 0.1},
        nonlin=e2_nn.ELU,
        nonlin_kwargs={'alpha': 0.1, 'inplace': True},
        return_skips=True,
        pool='conv'
    )

    print(model.out_types)
    print([x.shape for x in model(data)])
    print([type(x) for x in model(data)])
    print(model.compute_conv_feature_map_size((64, 128), 4))