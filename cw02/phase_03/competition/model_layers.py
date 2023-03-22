import logging
import keras as k

def define_dense_layers(parent,
                        layer_count:int=1,
                        dense_units:int=128,
                        activation:str='relu',
                        l1_regulization:float=0.0,
                        l2_regulization:float=0.0,
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
    l1_regulization : float, optional
        The L1 regulization, by default 0.0
    l2_regulization : float, optional
        The L2 regulization, by default 0.0
    dropout : float, optional
        The dropout rate, by default 0.0

    Returns
    -------
    keras.layers
        The last layer created.
    """
    assert layer_count > 0, 'layer_count must be greater than 0'

    # add the first layer
    layers = k.layers.Dense(
        units=dense_units,
        activation=activation,
        kernel_regularizer=k.regularizers.l1_l2(l1_regulization, l2_regulization))(parent)

    if dropout > 0:
        layers = k.layers.Dropout(dropout)(layers)

    # add additional layers if required
    for _ in range(layer_count - 1):
        layers= define_dense_layers(
            parent=layers,
            layer_count=1,
            dense_units=dense_units,
            activation=activation,
            l1_regulization=l1_regulization,
            l2_regulization=l2_regulization,
            dropout=dropout)

    return layers

def define_convnet_layers(parent,
                          block_count:int=1,
                          activation:str='relu',
                          cov_count:int=1,
                          channels:int=32,
                          kernel_size=(3, 3),
                          pool_size=(2, 2)):
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

    layer = parent
    for block in range(block_count):
        logging.info(f'block {block}')
        for cov in range(cov_count):
            logging.info(f'cov {cov}')
            layer = k.layers.Conv2D(
                filters=channels,
                kernel_size=kernel_size,
                padding='same',
                activation=activation)(layer)
            
        # add a pooling layer
        layer = k.layers.MaxPooling2D(pool_size=pool_size)(layer)

    # flatten for the last layer
    layer = k.layers.Flatten()(layer)

    return layer
                       