import keras as k
import mlflow

import competition.model_training as mt
import competition.model_layers as ml

def get_simple_dense_model(input_shape,
                           output_shape,
                           dense_layer_count:int=1,
                           dense_units:int=128,
                           dense_activation:str='relu',
                           dense_l1_regulization:float=0.0,
                           dense_l2_regulization:float=0.0,               
                           dense_dropout:float=0.0) -> k.Model:
    """
    Create a simple feed forward neural network consisting of dense layers.

    Parameters
    ----------
    input_shape : tuple
        The input shape.
    output_shape : int
        The output shape.
    dense_layer_count : int, optional
        The number of dense layers, by default 1
    dense_units : int, optional
        The number of units in each dense layer, by default 128
    dense_activation : str, optional
        The activation function, by default 'relu'
    dense_l1_regulization : float, optional
        The L1 regulization, by default 0.0
    dense_l2_regulization : float, optional
        The L2 regulization, by default 0.0
    dense_dropout : float, optional
        The dropout rate, by default 0.0

    Returns
    -------
    keras.Model
        The model.
    """
    # log model parameters
    mlflow.log_param('dense_layer_count', dense_layer_count)
    mlflow.log_param('dense_units', dense_units)
    mlflow.log_param('dense_activation', dense_activation)
    mlflow.log_param('dense_l1_regulization', dense_l1_regulization)
    mlflow.log_param('dense_l2_regulization', dense_l2_regulization)
    mlflow.log_param('dense_dropout', dense_dropout)
        
    # create the input layer
    input_layer = k.layers.Input(shape=input_shape)

    # create the dense layers
    dense_layers = ml.define_dense_layers(
        parent=input_layer,
        layer_count=dense_layer_count,
        dense_units=dense_units,
        activation=dense_activation,
        l1_regulization=dense_l1_regulization,
        l2_regulization=dense_l2_regulization,
        dropout=dense_dropout)
    
    # define the model output
    model_output = k.layers.Dense(output_shape, activation='sigmoid')(dense_layers)

    # create the model
    return k.Model(inputs=[input_layer], outputs=model_output)

def train_simple_dense(dataset:dict,
                       input_shape,
                       output_shape,
                       dense_layer_count:int=1,
                       dense_units:int=128,
                       dense_activation:str='relu',
                       dense_l1_regulization:float=0.0,
                       dense_l2_regulization:float=0.0,               
                       dense_dropout:float=0.2,
                       train_epochs:int=10,
                       train_batch_size:int=25,
                       train_optimizer:k.optimizers=k.optimizers.RMSprop(learning_rate=0.0001),
                       train_loss:str='binary_crossentropy',
                       train_metrics:list=['accuracy'],
                       train_class_weight:dict=None) -> k.Model:
    """
    Train a simple feed forward neural network consisting of dense layers.
    """
    # log model parameters
    mlflow.log_param('loss', train_loss)

    # create the model
    model = get_simple_dense_model(
        input_shape=input_shape,
        output_shape=output_shape,
        dense_layer_count=dense_layer_count,
        dense_units=dense_units,
        dense_activation=dense_activation,
        dense_l1_regulization=dense_l1_regulization,
        dense_l2_regulization=dense_l2_regulization,
        dense_dropout=dense_dropout)

    # plot the model architecture
    model.summary() 

    # train the model
    mt.train_and_test_model(
        model=model,
        X_train = dataset['train']['X'],
        y_train= dataset['train']['y'],
        X_val = dataset['val']['X'],
        y_val= dataset['val']['y'],
        X_test = dataset['test']['X'],
        y_test= dataset['test']['y'],
        epochs=train_epochs,
        batch_size=train_batch_size,
        optimizer=train_optimizer,
        loss=train_loss,
        metrics=train_metrics,
        class_weight=train_class_weight)

    return model      