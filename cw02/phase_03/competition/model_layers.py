"""
Define layer architectures that can be composed into models.s
"""

import logging
from typing import Tuple

from keras import layers
from keras import regularizers

def define_dense_layers(parent,
                        layer_count:int=1,
                        dense_units:int=128,
                        activation:str='relu',
                        l1_regularization:float=0.0,
                        l2_regularization:float=0.0,
                        dropout:float=0.0):
    """
    Create feed forward layers as per the parameters.

    Parameters
    ----------
    parent : keras.layers
        The parent layer.
    layer_count : int, optional
        The number of layers to create, by default 1
    dense_units : int, optional
        The number of units in each layer, by default 128
    activation : str, optional
        The activation function, by default 'relu'
    l1_regularization : float, optional
        The L1 regularization, by default 0.0
    l2_regularization : float, optional
        The L2 regularization, by default 0.0
    dropout : float, optional
        The dropout rate, by default 0.0

    Returns
    -------
    keras.layers
        The last layer created.
    """
    assert layer_count > 0, 'layer_count must be greater than 0'

    # add the first layer
    model_layers = layers.Dense(
        units=dense_units,
        activation=activation,
        kernel_regularizer=regularizers.l1_l2(l1_regularization, l2_regularization))(parent)

    if dropout > 0:
        model_layers = layers.Dropout(dropout)(model_layers)

    # add additional layers if required
    for _ in range(layer_count - 1):
        model_layers= define_dense_layers(
            parent=model_layers,
            layer_count=1,
            dense_units=dense_units,
            activation=activation,
            l1_regularization=l1_regularization,
            l2_regularization=l2_regularization,
            dropout=dropout)

    return model_layers

def define_convnet_layers(parent,
                          block_count:int,
                          activation:str,
                          cov_count:int,
                          channels:int,
                          kernel_size:Tuple[int, int],
                          pool_size:Tuple[int, int]):
    """
    Create convolutional layer blocks.
    
    Parameters
    ----------
    parent : keras.Model
        The parent model.
    block_count : int
        The number of convolutional layer blocks.
    activation : str, optional
        The activation function, by default 'relu'
    cov_count : int
        The number of convolutional layers in each block.
    channels : int
        The number of channels in each convolutional layer.
    kernel_size : tuple
        The size of the convolutional kernel.
    pool_size : tuple
        The size of the pooling kernel.

    Returns
    -------
    keras.layers
        The last layer created.
    """
    assert block_count > 0, 'block_count must be greater than 0'

    model_layer = parent
    for block in range(block_count):
        logging.info('block %s', block)
        for cov in range(cov_count):
            logging.info('cov %d', cov)
            model_layer = layers.Conv2D(
                filters=channels,
                kernel_size=kernel_size,
                padding='same',
                activation=activation)(model_layer)

        # add a pooling layer
        model_layer = layers.MaxPooling2D(pool_size=pool_size)(model_layer)

    # flatten for the last layer
    model_layer = layers.Flatten()(model_layer)

    return model_layer
