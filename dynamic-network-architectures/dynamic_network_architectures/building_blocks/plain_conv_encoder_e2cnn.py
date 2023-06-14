from typing import Tuple, List, Union, Type

import numpy as np
import torch
from torch import nn as torch_nn
from e2cnn import nn as e2_nn

from dynamic_network_architectures.building_blocks.simple_conv_blocks_e2cnn import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper_e2cnn import maybe_convert_scalar_to_list, get_matching_pooling_op


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

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        (isinstance(strides[s], (tuple, list))) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pooling_op(conv_op, pool_type=pool)())
                conv_stride = 1