from typing import Tuple, List, Union, Type

import numpy as np
import torch
from torch import nn as torch_nn
from e2cnn import nn as e2_nn

from dynamic_network_architectures.building_blocks.simple_conv_blocks_e2cnn import ConvertToTensor, ConvDropoutNormReLU, StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper_e2cnn import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder_e2cnn import PlainConvEncoder

class UNetDecoder(e2_nn.EquivariantModule):
    def __init__(self,
                 encoder: PlainConvEncoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 deep_supervision, 
                 nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, f"n_conv_per_stage must be of length {n_stages_encoder - 1}"

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            in_type_below = encoder.out_types[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(
                torch_nn.Sequential(
                    transpconv_op(
                        in_type=in_type_below,
                        scale_factor=stride_for_transpconv
                    ),
                    ConvDropoutNormReLU(
                        conv_op=encoder.conv_op,
                        in_type=in_type_below,
                        input_channels=encoder.output_channels[-s],
                        output_channels=input_features_skip,
                        kernel_size=encoder.kernel_sizes[-(s + 1)],
                        initial_stride=1,
                        conv_bias=encoder.conv_bias,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        dropout_op=encoder.dropout_op,
                        dropout_op_kwargs=encoder.dropout_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                        nonlin_first=nonlin_first
                    )
                )
            )

            stages.append(
                StackedConvBlocks(
                    num_convs=n_conv_per_stage[s-1],
                    conv_op=encoder.conv_op,
                    in_type=e2_nn.FieldType(encoder.gspace, 2*input_features_skip*[encoder.gspace.regular_repr]),
                    input_channels=2*input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    initial_stride=1,
                    conv_bias=encoder.conv_bias,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    dropout_op=encoder.dropout_op,
                    dropout_op_kwargs=encoder.dropout_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    nonlin_first=nonlin_first
                )
            )

            seg_layers.append(
                torch_nn.Sequential(
                    ConvertToTensor(
                        in_type=e2_nn.FieldType(encoder.gspace, input_features_skip*[encoder.gspace.regular_repr])
                    ),
                    torch_nn.Conv2d(
                        in_channels=input_features_skip,
                        out_channels=num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True
                    )
                )
            )

        self.transpconvs = torch_nn.ModuleList(transpconvs)
        self.stages = torch_nn.ModuleList(stages)
        self.seg_layers = torch_nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = e2_nn.tensor_directsum([x, skips[-(s+2)]])
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(

                    self.seg_layers[s](x)
                )
            elif s == (len(self.stages) - 1):
                seg_outputs.append(
                    self.seg_layers[-1](x)
                )
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size, order=4):
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append(
                [i // j for i,j in zip(input_size, self.encoder.strides[s])]
            )
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)], order=order)
            output += np.prod([self.encoder.output_channels[-(s+2)]*order, *skip_sizes[-(s+1)]], dtype=np.int64)
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        pass


if __name__ == '__main__':

    from e2cnn import gspaces

    # Test
    input_channels = 3
    data = torch.rand(1, input_channels, 256, 128)

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

    """
    output will be of shape:

    [torch.Size([1, 64, 128, 64]), 
    torch.Size([1, 64, 64, 32]), 
    torch.Size([1, 64, 32, 16]), 
    torch.Size([1, 64, 16, 8])]
    """

    decoder = UNetDecoder(
        encoder=model,
        num_classes=2,
        n_conv_per_stage=2,
        deep_supervision=True
    )

    print([type(x) for x in decoder(model(data))])
    print([x.shape for x in decoder(model(data))])