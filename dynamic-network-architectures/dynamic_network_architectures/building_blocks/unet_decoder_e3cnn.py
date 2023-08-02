import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple
from dynamic_network_architectures.building_blocks.simple_conv_blocks_e3cnn import StackedConvBlocks, ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder_e3cnn import PlainConvEncoder
from e3nn.o3 import Linear


class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: PlainConvEncoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        convs = []
        seg_layers_1 = []
        seg_layers_2 = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            if s == 1:
                input_transpconv = encoder.stages[-s][-1].irreps_out
            else:
                input_transpconv = stages[-1].irreps_out
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(nn.Upsample(scale_factor=stride_for_transpconv, mode='trilinear', align_corners=True))
            convs.append(ConvDropoutNormReLU(
                "SO3", 5, (1,1,1), input_transpconv, input_features_skip, 5, 1, True, encoder.conv_bias, 
                encoder.norm_op, encoder.norm_op_kwargs, encoder.dropout_op, encoder.dropout_op_kwargs, 
                encoder.nonlin, encoder.nonlin_kwargs, nonlin_first)
            )
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            input_irreps = convs[-1].gate.irreps_out + encoder.stages[-(s + 1)][-1].irreps_out
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], "SO3", 5, (1,1,1), input_irreps, 2*input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, True, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs, nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            out_layer = Linear(stages[-1].irreps_out, str(2*input_features_skip*7)+"x0e")
            outout_layer =  nn.Conv3d(
                            in_channels=2*input_features_skip*7,
                            out_channels=num_classes,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True
                        )
            seg_layers_1.append(out_layer)
            seg_layers_2.append(outout_layer)

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.convs = nn.ModuleList(convs)
        self.seg_layers_1 = nn.ModuleList(seg_layers_1)
        self.seg_layers_2 = nn.ModuleList(seg_layers_2)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = self.convs[s](x)
            x = torch.cat((x, skips[-(s+2)]), dim=1)
            x = self.stages[s](x)
            if self.deep_supervision:
                y = self.seg_layers_1[s](x.transpose(1, 4)).transpose(1, 4)
                seg_outputs.append(self.seg_layers_2[s](y))
            elif s == (len(self.stages) - 1):
                y = self.seg_layers_1[-1](x.transpose(1, 4)).transpose(1, 4)
                seg_outputs.append(self.seg_layers_2[-1](y))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, (self.encoder.strides[s],)*3)])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output
    

if __name__ == '__main__':

    input_channels = 3
    data = torch.rand(1, input_channels, 256, 128, 128)

    model = PlainConvEncoder(
        input_channels=input_channels,
        n_stages=4,
        features_per_stage=(1,2,4,8),
        conv_op=nn.Conv2d,
        kernel_sizes=5,
        strides=2,
        n_conv_per_stage=2,
        conv_bias=True,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        nonlin=nn.ReLU,
        nonlin_kwargs={'inplace': True},
        return_skips=True,
        pool='conv'
    )

    decoder = UNetDecoder(
        encoder=model,
        num_classes=2,
        n_conv_per_stage=2,
        deep_supervision=True
    )

    print([type(x) for x in decoder(model(data))])
    print([x.shape for x in decoder(model(data))])