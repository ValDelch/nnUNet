from typing import Type
import numpy as np
import torch.nn
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.instancenorm import _InstanceNorm
from BesselConv.BesselConv2d import BesselConv2d


def convert_dim_to_conv_op(dimension: int) -> Type[nn.Module]:
    """
    :param dimension: 1, 2 or 3
    :return: conv Class of corresponding dimension
    """
    if dimension == 2:
        return BesselConv2d
    else:
        raise ValueError("Unknown dimension. Only 2d is supported")


def convert_conv_op_to_dim(conv_op: Type[nn.Module]) -> int:
    """
    :param conv_op: conv class
    :return: dimension: 1, 2 or 3
    """
    if conv_op == BesselConv2d:
        return 2
    else:
        raise ValueError("Unknown dimension. Only 2d conv is supported. got %s" % str(conv_op))


def get_matching_pool_op(conv_op: Type[nn.Module] = None,
                         dimension: int = None,
                         adaptive=False,
                         pool_type: str = 'avg') -> Type[torch.nn.Module]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!
    :param conv_op:
    :param dimension:
    :param adaptive:
    :param pool_type: either 'avg' or 'max'
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    assert pool_type in ['avg', 'max'], 'pool_type must be either avg or max'
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [2], 'Dimension must be 2'

    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)

    if pool_type == 'avg':
        if adaptive:
            return nn.AdaptiveAvgPool2d
        else:
            return nn.AvgPool2d
    elif pool_type == 'max':
        if adaptive:
            return nn.AdaptiveMaxPool2d
        else:
            return nn.MaxPool2d


def get_matching_instancenorm(conv_op: Type[nn.Module] = None, dimension: int = None) -> Type[_InstanceNorm]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    if dimension is not None:
        assert dimension in [2], 'Dimension must be 2'
    return nn.InstanceNorm2d


def get_matching_convtransp(conv_op: Type[nn.Module] = None, dimension: int = None) -> Type[nn.Module]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [2], 'Dimension must be 2'
    return nn.Upsample


def get_matching_batchnorm(conv_op: Type[nn.Module] = None, dimension: int = None) -> Type[_BatchNorm]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [2], 'Dimension must be 2'
    return nn.BatchNorm2d


def get_matching_dropout(conv_op: Type[nn.Module] = None, dimension: int = None) -> Type[_DropoutNd]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    assert dimension in [2], 'Dimension must be 2'
    return nn.Dropout2d


def maybe_convert_scalar_to_list(conv_op, scalar):
    """
    useful for converting, for example, kernel_size=3 to [3, 3, 3] in case of nn.Conv3d
    :param conv_op:
    :param scalar:
    :return:
    """
    if not isinstance(scalar, (tuple, list, np.ndarray)):
        if conv_op == BesselConv2d:
            return [scalar] * 2
        else:
            raise RuntimeError("Invalid conv op: %s" % str(conv_op))
    else:
        return scalar


def get_default_network_config(dimension: int = 2,
                               nonlin: str = "Softsign",
                               norm_type: str = "bn") -> dict:
    """
    Use this to get a standard configuration. A network configuration looks like this:

    config = {'conv_op': torch.nn.modules.conv.Conv2d,
              'dropout_op': torch.nn.modules.dropout.Dropout2d,
              'norm_op': torch.nn.modules.batchnorm.BatchNorm2d,
              'norm_op_kwargs': {'eps': 1e-05, 'affine': True},
              'nonlin': torch.nn.modules.activation.ReLU,
              'nonlin_kwargs': {'inplace': True}}

    There is no need to use get_default_network_config. You can create your own. Network configs are a convenient way of
    setting dimensionality, normalization and nonlinearity.

    :param dimension: integer denoting the dimension of the data. 1, 2 and 3 are accepted
    :param nonlin: string (ReLU or LeakyReLU)
    :param norm_type: string (bn=batch norm, in=instance norm)
    torch.nn.Module
    :return: dict
    """
    config = {}
    config['conv_op'] = convert_dim_to_conv_op(dimension)
    config['dropout_op'] = get_matching_dropout(dimension=dimension)
    if norm_type == "bn":
        config['norm_op'] = get_matching_batchnorm(dimension=dimension)
    elif norm_type == "in":
        config['norm_op'] = get_matching_instancenorm(dimension=dimension)

    config['norm_op_kwargs'] = None # this will use defaults

    if nonlin == "LeakyReLU":
        config['nonlin'] = nn.LeakyReLU
        config['nonlin_kwargs'] = {'negative_slope': 1e-2, 'inplace': True}
    elif nonlin == "ReLU":
        config['nonlin'] = nn.ReLU
        config['nonlin_kwargs'] = {'inplace': True}
    elif nonlin == "Softsign":
        config['nonlin'] = nn.Softsign
        config['nonlin_kwargs'] = {}
    else:
        raise NotImplementedError('Unknown nonlin %s. Only "LeakyReLU", "ReLU" and Softsign are supported for now' % nonlin)

    return config


if __name__ == "__main__":

    # Test

    print("Convert dim to conv:", convert_dim_to_conv_op(2))
    print("Convert conv to dim:", convert_conv_op_to_dim(BesselConv2d))
    print("Get matching pool op avg", get_matching_pool_op(dimension=2, pool_type='avg'))
    print("Get matching pool op max", get_matching_pool_op(dimension=2, pool_type='max'))
    print("Get matching bn", get_matching_batchnorm(dimension=2))
    print("Get matching ConvTransp", get_matching_convtransp(dimension=2))
    print("Get matching dropout", get_matching_dropout(dimension=2))
    print("Get network config", get_default_network_config(dimension=2, nonlin="LeakyReLU", norm_type="bn"))