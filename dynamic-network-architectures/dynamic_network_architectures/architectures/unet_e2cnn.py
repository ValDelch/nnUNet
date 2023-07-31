from typing import Tuple, List, Union, Type

import numpy as np
import torch
from torch import nn as torch_nn
from escnn import nn as e2_nn

from dynamic_network_architectures.building_blocks.plain_conv_encoder_e2cnn import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder_e2cnn import UNetDecoder
#from dynamic_network_architectures.building_blocks.unet_decoder_for_e2cnn import UNetDecoder
from dynamic_network_architectures.building_blocks.helper_e2cnn import convert_conv_op_to_dim


class PlainConvUNet(torch_nn.Module):
    def __init__(self,
                 gspace: Type[e2_nn.FieldType],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[e2_nn.EquivariantModule],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[e2_nn.EquivariantModule]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[e2_nn.EquivariantModule]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[e2_nn.EquivariantModule]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        super().__init__()

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, f"n_conv_per_stage must have as many entries as we have " \
                                                    f"resolution stages. here: {n_stages}. " \
                                                    f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        
        self.encoder = PlainConvEncoder(gspace, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, 
                                        nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first)
        
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first=nonlin_first)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
    
    def compute_conv_feature_map_size(self, input_size, order=4):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "input_size must have as many entries as we have spatial dimensions"
        return self.encoder.compute_conv_feature_map_size(input_size, order=order) + self.decoder.compute_conv_feature_map_size(input_size, order=order)
    

if __name__ == '__main__':

    from escnn import gspaces

    # Test 2D

    data = torch.rand((1, 4, 512, 512))
    r2_act = gspaces.rot2dOnR2(N=4)

    model = PlainConvUNet(r2_act, 4, 6, (8, 16, 63, 64, 128, 128), e2_nn.R2Conv, 5, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                          (2, 2, 2, 2, 2), False, e2_nn.InnerBatchNorm, None, None, None, e2_nn.ReLU, deep_supervision=True)

    print(model.compute_conv_feature_map_size(data.shape[2:], order=4))
    print([x.shape for x in model(data)])

    # Test 3D

    data = torch.rand((1, 3, 64, 32, 32))
    #r2_act = gspaces.octaOnR3()
    r2_act = gspaces.rot3dOnR3()

    model = PlainConvUNet(r2_act, 3, 4, (8, 16, 63, 64), e2_nn.R3Conv, 5, (1, 2, 2, 2), (2, 2, 2, 2), 4,
                          (2, 2, 2), False, e2_nn.InnerBatchNorm, None, None, None, e2_nn.ReLU, deep_supervision=True)

    print(model.compute_conv_feature_map_size(data.shape[2:], order=4))
    print([x.shape for x in model(data)])