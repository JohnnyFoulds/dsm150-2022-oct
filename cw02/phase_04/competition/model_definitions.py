import tensorflow as tf
from keras import layers
import keras as k
from keras.optimizers import Adam
import keras_tuner as kt
from typing import List

import competition.model_training as mt
import competition.model_layers as ml

def compile_model(model: k.Model,
                  optimizer,
                  loss: str,
                  metrics: List[str]) -> None:
    
    # compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

def get_activation_layer(activation_name):
    if activation_name == 'LeakyReLU':
        return layers.LeakyReLU(alpha=0.3)
    elif activation_name == 'PReLU':
        return layers.PReLU()
    elif activation_name == 'Swish':
        return layers.Activation('swish')
    else:
        return activation_name
