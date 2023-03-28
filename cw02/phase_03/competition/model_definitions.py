import tensorflow as tf
import keras as k
import kerastuner as kt
import mlflow
import logging
from functools import partial

import competition.model_training as mt
import competition.model_layers as ml

def compile_model(model,
                  optimizer:k.optimizers,
                  loss:str,
                  metrics:list) -> None:
    
    # compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

def get_simple_dense_model(input_shape,
                           output_shape,
                           dense_layer_count:int=1,
                           dense_units:int=128,
                           dense_activation:str='relu',
                           dense_l1_regulization:float=0.0,
                           dense_l2_regulization:float=0.0,               
                           dense_dropout:float=0.0,
                           compile:bool=False,
                           optimizer:k.optimizers=None,
                           learning_rate:float=0.001,
                           loss:str=None,
                           metrics:list=None) -> k.Model:
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
    logging.info('Creating simple dense model')

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
    model = k.Model(inputs=[input_layer], outputs=model_output)

    # compile the model
    if compile:
        compile_model(
            model=model,
            optimizer=optimizer(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics)

    return model

def get_activation_layer(activation_name):
    if activation_name == 'LeakyReLU':
        return k.layers.LeakyReLU(alpha=0.3)
    elif activation_name == 'PReLU':
        return k.layers.PReLU()
    elif activation_name == 'Swish':
        return k.layers.swish
    else:
        return activation_name


def get_simple_dense_model_wrapper(hp,
                                   define_tune_parameters:callable,
                                   input_shape,
                                   output_shape,
                                   loss,                                
                                   *args,
                                   **kwargs) -> k.Model:
    """
    This function takes the extra hyper parameter object that are required
    for the build function.
    """
    # define the training parameters
    define_tune_parameters(hp)

    for kwarg in kwargs:
        if isinstance(kwargs[kwarg], str):
            kwargs[kwarg] = hp.get(kwargs[kwarg])

    # update dense_activation
    kwargs['dense_activation'] = get_activation_layer(kwargs['dense_activation'])

    return get_simple_dense_model(
        input_shape=input_shape,
        output_shape=output_shape,
        loss=loss,
        compile=True,
        *args, **kwargs)

class TrailCallback(k.callbacks.Callback):
    def on_train_begin(self, logs=None):
        logging.info('on_train_begin')


    def on_train_end(self, logs=None):
        logging.info('on_train_end')

def tune_simple_dense_model(define_tune_parameters:callable,
                            dataset:dict,
                            max_trials:int,
                            input_shape,
                            output_shape,
                            dense_layer_count:str='dense_layer_count',
                            dense_units:str='dense_units',
                            dense_activation:str='dense_activation',
                            dense_l1_regulization:float='dense_l1_regulization',
                            dense_l2_regulization:float='dense_l2_regulization',               
                            dense_dropout:float='dropout_rate',
                            train_epochs:int=10,
                            train_batch_size:int=25,
                            train_optimizer:k.optimizers=k.optimizers.RMSprop,
                            train_learning_rate:str='learning_rate',
                            train_loss:str='binary_crossentropy',
                            train_metrics:list=['accuracy'],
                            train_class_weight:dict=None,
                            tuner_type=kt.tuners.RandomSearch,
                            tune_objective:str='val_accuracy',
                            tune_direction:str='max') -> k.Model:
    """
    Find the optimal hyper parameters using the KerasTuner API.
    """
    # create the partial function to build the model
    build_model = partial(get_simple_dense_model_wrapper,
        define_tune_parameters=define_tune_parameters,
        input_shape=input_shape,
        output_shape=output_shape,
        loss=train_loss,
        dense_layer_count=dense_layer_count,
        dense_units=dense_units,
        dense_activation=dense_activation,
        dense_l1_regulization=dense_l1_regulization,
        dense_l2_regulization=dense_l2_regulization,
        dense_dropout=dense_dropout,
        optimizer=train_optimizer,
        learning_rate=train_learning_rate,
        metrics=train_metrics)
    
    # create the callback for testing
    test_callback = mt.TestModelCallback(
        X_train=dataset['train']['X'],
        y_train=dataset['train']['y'],
        X_val=dataset['val']['X'],
        y_val=dataset['val']['y'],
        X_test=dataset['test']['X'],
        y_test=dataset['test']['y'],
        show_plots=True)    

    # define CustomSearch class
    class CustomSearch(tuner_type):
        def on_trial_begin(self, trial):
            logging.info('on_trial_begin')
            mlflow.keras.autolog()
            mlflow.start_run(nested=True)

            super(CustomSearch, self).on_trial_begin(trial)

        def on_trial_end(self, trial):
            logging.info('on_trial_end')
            mlflow.end_run()

            super(CustomSearch, self).on_trial_end(trial)

    with mlflow.start_run() as run:
        tuner = CustomSearch(
            build_model,
            objective=kt.Objective(tune_objective, direction=tune_direction),
            max_trials=max_trials,
            executions_per_trial=1,
            overwrite=True)
        
        run_id = run.info.run_id
    mlflow.end_run()
    mlflow.delete_run(run_id)            

    # start the search
    with mlflow.start_run():
        mlflow.keras.autolog()

        # search for the best hyperparameters
        tuner.search(
            dataset['train']['X'],
            dataset['train']['y'],
            validation_data=(dataset['val']['X'], dataset['val']['y']),
            epochs=train_epochs,
            batch_size=train_batch_size,
            class_weight=train_class_weight,
            callbacks=[test_callback, tf.keras.callbacks.EarlyStopping(patience=100)])
            #callbacks=[test_callback, tf.keras.callbacks.EarlyStopping(patience=2)])
        
        # log the best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0].values
        mlflow.log_params(best_hp)
        print(best_hp)

        # Retrieve the best model, evaluate it, and log the metrics
        best_model = tuner.get_best_models()[0]
        val_loss, val_objective = best_model.evaluate(dataset['val']['X'], dataset['val']['y'])
        mlflow.log_metric("val_loss", val_loss)
        mlflow.log_metric(tune_objective, val_objective)

    mlflow.end_run()    


    return best_model


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