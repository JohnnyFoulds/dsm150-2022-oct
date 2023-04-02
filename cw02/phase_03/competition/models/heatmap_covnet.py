"""
This module contains the definition using the heatmap as a feature and
using a Convolutional Neural Network to classify the data.
"""

import logging
from functools import partial
from typing import Callable, Dict, Optional, Tuple, List, Type
import ast
import mlflow
import mlflow.keras

import tensorflow as tf
import keras as k
from keras import layers
from keras_tuner.engine import tuner as tuner_module
from keras_tuner import Objective

import competition.model_definitions as mm
import competition.model_layers as ml
import competition.model_training as mt

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
                   heatmap_dataset:Dict,
                   feature_dataset:Dict,
                   max_trials:int,
                   train_epochs:int,
                   train_batch_size:int,
                   train_optimizer:Optional[Callable],
                   tuner_type:Type[tuner_module.Tuner],
                   tune_objective:str,
                   tune_direction:str,
                   train_class_weight:Optional[Dict]=None) -> k.Model:
        """
        Find the optimal hyper parameters using the KerasTuner API.
        """
        # create the partial function to build the model
        build_model = partial(
            self.get_model_wrapper,
            define_tune_parameters=define_tune_parameters,
            optimizer=train_optimizer)

        # create the callback for testing
        test_callback = mt.TestModelCallback(
            X_train=[heatmap_dataset['train']['X'], feature_dataset['train']['X']],
            y_train=feature_dataset['train']['y'],
            X_val=[heatmap_dataset['val']['X'], feature_dataset['val']['X']],
            y_val=feature_dataset['val']['y'],
            X_test=[heatmap_dataset['test']['X'], feature_dataset['test']['X']],
            y_test=feature_dataset['test']['y'],
            show_plots=True)

        class CustomSearch(tuner_type):
            """A wrapper around the KerasTuner API to enable nested experiments."""
            def on_trial_begin(self, trial):
                """Start a nested run at the beginning of each trial."""
                mlflow.keras.autolog()
                mlflow.start_run(nested=True)

                super(CustomSearch, self).on_trial_begin(trial)

            def on_trial_end(self, trial):
                """End the nested run at the end of each trial."""
                mlflow.end_run()
                super(CustomSearch, self).on_trial_end(trial)

        with mlflow.start_run() as run:
            tuner = CustomSearch(
                build_model,
                objective=Objective(tune_objective, direction=tune_direction),
                max_trials=max_trials,
                executions_per_trial=1,
                directory='./untitled_project/heatmap_covnet',
                overwrite=True)

            run_id = run.info.run_id

        mlflow.end_run()
        mlflow.delete_run(run_id)

        # start the search
        with mlflow.start_run():
            mlflow.keras.autolog()

            # search for the best hyperparameters
            tuner.search(
                [heatmap_dataset['train']['X'], feature_dataset['train']['X']],
                feature_dataset['train']['y'],
                validation_data=( \
                    [heatmap_dataset['val']['X'], feature_dataset['val']['X']],
                    feature_dataset['val']['y']),
                epochs=train_epochs,
                batch_size=train_batch_size,
                class_weight=train_class_weight,
                callbacks=[test_callback, tf.keras.callbacks.EarlyStopping(patience=50)])

            # log the best hyperparameters
            best_hp = tuner.get_best_hyperparameters()[0].values
            mlflow.log_params(best_hp)
            print(best_hp)

            # Retrieve the best model, evaluate it, and log the metrics
            best_model = tuner.get_best_models()[0]
            val_loss, val_objective = best_model.evaluate( \
                [heatmap_dataset['val']['X'], feature_dataset['val']['X']],
                feature_dataset['val']['y'])
            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric(tune_objective, val_objective)

        mlflow.end_run()

        return best_model
