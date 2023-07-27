from typing import Type
import numpy as np

from escnn import nn
from escnn.nn import EquivariantModule 


def convert_dim_to_conv_op(dimension: int) -> Type[EquivariantModule]:
    if dimension == 2:
        return nn.R2Conv
    elif dimension == 3:
        return nn.R3Conv
    else:
        raise ValueError('Equivariant models are just compatible with 2D or 3D images')


def convert_conv_op_to_dim(conv_op: Type[EquivariantModule]):
    if conv_op == nn.R2Conv:
        return 2
    elif conv_op == nn.R3Conv:
        return 3
    else:
        raise ValueError('Equivariant models are just compatible with 2D or 3D images')


def get_matching_pool_op(conv_op: Type[EquivariantModule] = None,
                         dimension: int = None,
                         adaptative=False,
                         pool_type: str = 'avg') -> Type[EquivariantModule]:
    
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    assert pool_type in ['avg', 'max'], 'pool_type must be either avg or max'

    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)

    assert dimension in [2,3], 'Dimension must be 2 or 3'

    if dimension == 2:
        if pool_type == 'avg':
            if adaptative:
                return nn.PointwiseAdaptiveAvgPool2D
            else:
                return nn.PointwiseAvgPool2D
        elif pool_type == 'max':
            if adaptative:
                return nn.PointwiseAdaptiveMaxPool2D
            else:
                return nn.PointwiseMaxPool2D
    elif dimension == 3:
        if pool_type == 'avg':
            if adaptative:
                return nn.PointwiseAdaptiveAvgPool3D
            else:
                return nn.PointwiseAvgPool3D
        elif pool_type == 'max':
            if adaptative:
                return nn.PointwiseAdaptiveMaxPool3D
            else:
                return nn.PointwiseMaxPool3D



def get_matching_instancenorm(conv_op: Type[EquivariantModule] = None,
                              dimension: int = None) -> Type[EquivariantModule]:
    
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    
    raise NotImplementedError('InstanceNorm is not implemented yet for equivariant models')


def get_matching_batchnorm(conv_op: Type[EquivariantModule] = None,
                           dimension: int = None) -> Type[EquivariantModule]:
    
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)

    assert dimension in [2,3], 'Dimension must be 2 or 3'

    return nn.InnerBatchNorm


def get_matching_convtransp(conv_op: Type[EquivariantModule] = None,
                            dimension: int = None) -> Type[EquivariantModule]:
    """
    The e2cnn library implements an R2ConvTranspose module.
    However, they highlight that it may introduce artifacts in the final equivariant
    and they recomment to use a simple upsampling combined with R2Conv instead.
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)

    assert dimension in [2,3], 'Dimension must be 2 or 3'

    if dimension == 2:
        return nn.R2Upsampling
    elif dimension == 3:
        return nn.R3Upsampling


def get_matching_dropout(conv_op: Type[EquivariantModule] = None,
                         dimension: int = None) -> Type[EquivariantModule]:
    
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    assert dimension in [2,3], 'Dimension must be 2 or 3'

    return nn.PointwiseDropout


def maybe_convert_scalar_to_list(conv_op, scalar):

    if not isinstance(scalar, (tuple, list, np.ndarray)):
        dimension = convert_conv_op_to_dim(conv_op)
        assert dimension in [2,3], 'Dimension must be 2 or 3'
        if dimension == 2:
            return [scalar] * 2
        elif dimension == 3:
            return [scalar] * 3
    else:
        return scalar
    

def get_default_network_config(dimension: int = 2,
                               nonlin: str = "ReLU",
                               norm_type: str = "bn") -> dict:
    """
    LeakyReLU is not implemented yet in e2cnn so it is replaced by ELU for now.
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
        config['nonlin'] = nn.ELU
        config['nonlin_kwargs'] = {'alpha': 0.1, 'inplace': True}
    elif nonlin == "ReLU":
        config['nonlin'] = nn.ReLU
        config['nonlin_kwargs'] = {'inplace': True}
    else:
        raise NotImplementedError('Unknown nonlin %s. Only "LeakyReLU" and "ReLU" are supported for now' % nonlin)

    return config


if __name__ == "__main__":

    # Test

    print("Convert dim to conv:", convert_dim_to_conv_op(3))
    print("Convert conv to dim:", convert_conv_op_to_dim(nn.R3Conv))
    print("Get matching pool op avg", get_matching_pool_op(dimension=3, pool_type='avg'))
    print("Get matching pool op max", get_matching_pool_op(dimension=3, pool_type='max'))
    print("Get matching bn", get_matching_batchnorm(dimension=3))
    print("Get matching ConvTransp", get_matching_convtransp(dimension=3))
    print("Get matching dropout", get_matching_dropout(dimension=3))
    print("Get network config", get_default_network_config(dimension=3, nonlin="LeakyReLU", norm_type="bn"))

