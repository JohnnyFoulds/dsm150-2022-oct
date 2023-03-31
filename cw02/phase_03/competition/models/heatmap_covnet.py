"""
This module contains the definition using the heatmap as a feature and
using a Convolutional Neural Network to classify the data.
"""

import logging
from typing import Callable, Dict, Optional, Tuple, List
import ast
import mlflow

import keras as k
from keras import layers
from keras import optimizers

from keras_tuner import tuners
from keras_tuner.engine import tuner as tuner_module

import competition.model_definitions as mm
import competition.model_layers as ml

class HeatmapCovnetModel():
    """
    This class contains the definition using the heatmap as a feature and
    using a Convolutional Neural Network to classify the data.
    """

    def __init__(self,
                 input_shape:int,
                 output_shape:int,
                 heatmap_shape:Tuple[int, int, int],
                 loss:str,
                 metrics:List) -> None:
        """
        Initialize the class.

        Parameters
        ----------
        input_shape : int
            The input shape.
        output_shape : int
            The output shape.
        heatmap_shape : tuple
            The heatmap shape.
        loss : str
            The loss function.
        metrics : list
            The metrics.

        Returns
        -------
        None
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.heatmap_shape = heatmap_shape
        self.loss = loss
        self.metrics = metrics

    def get_model(self,
                  covnet_block_count:int,
                  covnet_activation:str,
                  covnet_cov_count:int,
                  covnet_channels:int,
                  covnet_kernel_size:Tuple[int, int],
                  covnet_pool_size:Tuple[int, int],
                  dense_layer_count:int,
                  dense_units:int,
                  dense_activation:str,
                  dense_l1_regularization:float,
                  dense_l2_regularization:float,
                  dense_dropout:float,
                  compile_model:bool,
                  optimizer:Optional[Callable],
                  learning_rate:float) -> k.Model:
        """
        Create a convolutional neural network using the heatmap as a feature.

        Parameters
        ----------
        covnet_block_count : int
            The number of convolutional blocks.
        covnet_activation : str
            The activation function.
        covnet_cov_count : int
            The number of convolutions in each block.
        covnet_channels : int
            The number of channels in each convolution.
        covnet_kernel_size : tuple
            The kernel size.
        covnet_pool_size : tuple
            The pooling size.
        dense_layer_count : int
            The number of dense layers.
        dense_units : int
            The number of units in each dense layer.
        dense_activation : str
            The activation function.
        dense_l1_regularization : float
            The L1 regularization.
        dense_l2_regularization : float
            The L2 regularization.
        dense_dropout : float
            The dropout rate.
        compile : bool
            Whether to compile the model.
        optimizer : Callable
            The optimizer.
        learning_rate : float
            The learning rate.
        
        Returns
        -------
        keras.Model
            The model.
        """
        logging.info('Creating a heatmap CNN model')

        # log the model parameters
        mlflow.log_param('covnet_block_count', covnet_block_count)
        mlflow.log_param('covnet_activation', covnet_activation)
        mlflow.log_param('covnet_cov_count', covnet_cov_count)
        mlflow.log_param('covnet_channels', covnet_channels)
        mlflow.log_param('covnet_kernel_size', covnet_kernel_size)
        mlflow.log_param('covnet_pool_size', covnet_pool_size)
        mlflow.log_param('dense_layer_count', dense_layer_count)
        mlflow.log_param('dense_units', dense_units)
        mlflow.log_param('dense_activation', dense_activation)
        mlflow.log_param('dense_l1_regularization', dense_l1_regularization)
        mlflow.log_param('dense_l2_regularization', dense_l2_regularization)
        mlflow.log_param('dense_dropout', dense_dropout)

        # create the input layers
        input_layer = layers.Input(shape=(self.input_shape,), name='features')
        input_layer_heatmap = layers.Input(shape=self.heatmap_shape, name='heatmap')

        # create the convolutional layers
        covnet_layers = ml.define_convnet_layers(
            parent=input_layer_heatmap,
            block_count=covnet_block_count,
            activation=covnet_activation,
            cov_count=covnet_cov_count,
            channels=covnet_channels,
            kernel_size=covnet_kernel_size,
            pool_size=covnet_pool_size)

        # create the dense layers
        dense_layers = ml.define_dense_layers(
            parent=input_layer,
            layer_count=dense_layer_count,
            dense_units=dense_units,
            activation=dense_activation,
            l1_regularization=dense_l1_regularization,
            l2_regularization=dense_l2_regularization,
            dropout=dense_dropout)

        # concatenate image branch with flat features
        concat = layers.Concatenate()([covnet_layers, dense_layers])

        # define the model output
        model_output = layers.Dense(self.output_shape, activation='sigmoid')(concat)

        # create the model
        model = k.Model(inputs=[input_layer_heatmap, input_layer], outputs=model_output)

        # compile the model
        if compile_model and (optimizer is not None):
            mm.compile_model(
                model=model,
                optimizer=optimizer(learning_rate=learning_rate),
                loss=self.loss,
                metrics=self.metrics)

        return model
    
    def get_model_wrapper(self,
                          hp,
                          define_tune_parameters,
                          optimizer:Optional[Callable]) -> k.Model:
        """
        This function takes the extra hyper parameter object that are required
        for the build function.
        """
        # define the training parameters
        define_tune_parameters(hp)

        # get the model
        return self.get_model(
            covnet_block_count=hp['covnet_block_count'],
            covnet_activation=mm.get_activation_layer(hp['covnet_activation']),
            covnet_cov_count=hp['covnet_cov_count'],
            covnet_channels=hp['covnet_channels'],
            covnet_kernel_size=ast.literal_eval(hp['covnet_kernel_size']),
            covnet_pool_size=ast.literal_eval(hp['covnet_pool_size']),
            dense_layer_count=hp['dense_layer_count'],
            dense_units=hp['dense_units'],
            dense_activation=mm.get_activation_layer(hp['dense_activation']),
            dense_l1_regularization=hp['dense_l1_regularization'],
            dense_l2_regularization=hp['dense_l2_regularization'],
            dense_dropout=hp['dense_dropout'],
            compile_model=True,
            optimizer=optimizer,
            learning_rate=hp['learning_rate'])
    

    def tune_model(self,
                   define_tune_parameters,
                   dataset:Dict,
                   max_trials:int,
                   train_epochs:int,
                   train_batch_size:int,
                   train_optimizer:Optional[Callable],
                   tuner_type:tuner_module.Tuner,
                   tune_objective:str,
                   tune_direction:str,
                   train_class_weight:Optional[Dict]=None) -> k.Model:
        pass