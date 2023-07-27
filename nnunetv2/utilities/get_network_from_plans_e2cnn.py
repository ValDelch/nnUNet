from dynamic_network_architectures.architectures.unet_for_e2cnn import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper_e2cnn import get_matching_batchnorm, convert_dim_to_conv_op
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from escnn import gspaces
from escnn import nn as e2_nn


def get_network_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = configuration_manager.UNet_class_name
    mapping = {
        'PlainConvUNet': PlainConvUNet,
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_batchnorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': e2_nn.ELU, 'nonlin_kwargs': {'alpha': 0.1, 'inplace': True},
        }
    }
    assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                              'into either this ' \
                                                              'function (get_network_from_plans) or ' \
                                                              'the init of your nnUNetModule to accomodate that.'
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }
    # network class name!!
    if dim == 2:
        gspace = gspaces.rot2dOnR2(N=4)
    elif dim == 3:
        gspace = gspaces.octaOnR3()

    model = network_class(
        gspace=gspace,
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    #model.apply(InitWeights_He(1e-2))
    #if network_class == ResidualEncoderUNet:
    #    model.apply(init_last_bn_before_add_to_0)
    return model
