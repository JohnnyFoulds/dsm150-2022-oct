import numpy as np
from typing import Optional, List, Tuple
import logging

import mlflow

import keras as k
from tensorflow.keras.callbacks import History

from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
from IPython.display import Markdown

# Initialize an empty list to store the lines of markdown
_markdown_lines = []

def mprint(text):
    global _markdown_lines
    _markdown_lines.append(text)

def mflush():
    global _markdown_lines
    markdown_string = "\n".join(_markdown_lines)
    display(Markdown(markdown_string))
    # Clear the lines after displaying the markdown
    _markdown_lines = []

def plot_loss(history:History,
              figsize:Tuple[int, int] = (5, 3)) -> None:
    """
    Plot the loss and validation loss.

    Parameters
    ----------
    history : keras.callbacks.History
        The history of the model training.
    figsize : Tuple[int, int]
        The size of the figure to plot.

    Returns
    -------
    None
    """
    metric = None
    if 'accuracy' in history.history.keys():
        metric = 'accuracy'
    elif 'f1_score' in history.history.keys():
        metric = 'f1_score'
    else:
        print(history.history.keys())
        raise Exception('No metric found in history!')


    epochs = range(1, len(history.history[metric]) + 1)

    # summarize history for loss
    plt.figure(figsize=figsize)
    plt.plot(epochs, history.history['loss'])
    
    if ('val_loss' in history.history):
        plt.plot(epochs, history.history['val_loss'])
        plt.legend(['Training loss', 'Validation loss'], loc='upper left')
        plt.title('Training and validation loss')
    else:
        plt.title('Training loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()

def plot_accuracy(history:History,
                  figsize:Tuple[int, int] = (5, 3)) -> None:
    """
    Plot the accuracy and validation accuracy.

    Parameters
    ----------
    history : keras.callbacks.History
        The history of the model training.
    figsize : Tuple[int, int]
        The size of the figure to plot.

    Returns
    -------
    None
    """
    metric = None
    if 'accuracy' in history.history.keys():
        metric = 'accuracy'
    elif 'f1_score' in history.history.keys():
        metric = 'f1_score'
    else:
        print(history.history.keys())
        raise Exception('No metric found in history!')

    epochs = range(1, len(history.history[metric]) + 1)

    # summarize history for accuracy
    plt.figure(figsize=figsize)
    plt.plot(epochs, history.history[metric])

    if (f'val_{metric}' in history.history):
        plt.plot(epochs, history.history[f'val_{metric}'])
        plt.legend([f'Training {metric}', f'Validation {metric}'], loc='upper left')
        plt.title(f'Training and validation {metric}')
    else:
        plt.title(f'Training {metric}')

    plt.xlabel('Epochs')
    plt.ylabel(metric)

    plt.show()

def optimize_f1(y_true: np.ndarray,
                y_score: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Optimize the F1 score.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_score : np.ndarray
        The predicted labels.

    Returns
    -------
    Tuple[float, float, float]
        The optimized threshold, precision, and recall.
    """
    best_f1 = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0

    for threshold in np.arange(0, 1, 0.01):
        y_pred = (y_score > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return best_threshold, best_precision, best_recall, best_f1

def test_model(
        model,
        history: History,
        X: np.ndarray,
        y: np.ndarray,
        q: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        q_test: np.ndarray,
        show_plots: bool = True) -> Tuple[float, str, dict]:
    """
    Test the model based on the test data.

    Parameters
    ----------
    model : keras.models
        The model to test.
    history : keras.callbacks.History
        The history of the training.
    X : np.ndarray
        The training and validation features combined.
    y: np.ndarray
        The training and validation labels combined.
    q : np.ndarray
        The training and validation question numbers combined.
    X_test : np.ndarray
        The test data.
    y_test : np.ndarray
        The test labels.
    q_test : np.ndarray
        The test question numbers.

    Returns
    -------
    float
        The optimized threshold for the best F1 score.
    """
    metrics = {}

    if show_plots:
        plot_loss(history)
        plot_accuracy(history)

    # score the test data
    y_test_score = model.predict(X_test)

    # score the train and validation data
    y_score = model.predict(X)

    # display the results with a threshold of 0.5
    threshold = 0.5

    mprint('#### Threshold: 0.5')
    mprint('```')
    report = classification_report(y_test, y_test_score > threshold, zero_division=1)
    mprint(report)
    mprint('```')
    mflush()

    # calculate the f1 score
    _, _, f1, _ = precision_recall_fscore_support(
                y_test, 
                y_test_score > threshold, 
                average='macro',
                zero_division=1)    


    # save the metrics
    metrics['threshold'] = threshold
    metrics['classification_report'] = report
    metrics['f1'] = f1

    # show the confusion matrix
    if show_plots:
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true=y_test, 
            y_pred= y_test_score > threshold,
            cmap=plt.cm.Blues,
            normalize='true')
        metrics['confusion_matrix'] = plt.gcf()
        plt.show()

    # optimize the threshold for the best F1 score
    threshold, _, _, _ = optimize_f1(y, y_score)

    mprint(f'#### Optimal Threshold: {threshold:.2f}')
    mprint('```')
    report = classification_report(y_test, y_test_score > threshold, zero_division=1)
    mprint(report)
    mprint('```')
    mflush()

    # calculate the f1 score
    _, _, f1, _ = precision_recall_fscore_support(
                y_test, 
                y_test_score > threshold, 
                average='macro',
                zero_division=1)  
    
    # save the metrics
    metrics['threshold_optimized'] = threshold
    metrics['classification_report_optimized'] = report
    metrics['f1_optimized'] = f1

    # show the confusion matrix
    if show_plots:
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true=y_test, 
            y_pred= y_test_score > threshold,
            cmap=plt.cm.Blues,
            normalize='true')
        metrics['confusion_matrix_optimized'] = plt.gcf()
        plt.show()

    # try to optimize for each question
    # unfortunately, this doesn't work well
    if False:
        if q is not None:
            mprint(f'#### Optimizing Individial Questions')
            thresholds = optimize_question_f1(y, y_score, q)
            mprint('```')
            #mprint(report)
            mprint('```')
            mflush()

            display(thresholds)

    return threshold, report, metrics

class TestModelCallback(k.callbacks.Callback):
    """
    A callback to test the model after training completes.
    """
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 show_plots: bool):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.show_plots = show_plots

    def on_train_end(self, logs=None):
        logging.info('TestModelCallback::Testing the model')
        try:
            # combine the training and validation sets for testing
            if isinstance(self.X_train, list):
                X_combined = [np.concatenate((self.X_train[i], self.X_val[i]), axis=0) for i in range(len(self.X_train))]
                y_combined = np.concatenate((self.y_train, self.y_val), axis=0)
            else:
                X_combined = np.concatenate((self.X_train, self.X_val), axis=0)
                y_combined = np.concatenate((self.y_train, self.y_val), axis=0)

            # test the model
            threshold, report, metrics = test_model(
                self.model, 
                self.model.history,
                X=X_combined, 
                y=y_combined,
                q=None,
                X_test=self.X_test, 
                y_test=self.y_test, 
                q_test=None,
                show_plots=self.show_plots)
            
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    mlflow.log_dict(value, f'metrics/{metric}')
                elif isinstance(value, str):
                    mlflow.log_text(value, f'metrics/{metric}.txt')
                elif isinstance(value, plt.Figure):
                    mlflow.log_figure(value, f'metrics/{metric}.png')
                else:
                    mlflow.log_metric(metric, value)
                    
        except Exception as e:
            logging.error(f'Error testing the model: {e}')
            mlflow.log_text(f'Error testing the model: {e}', 'error.txt')
            

def train_model(
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val : np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int,
        batch_size: int,
        optimizer,
        loss: str,
        metrics: list,
        class_weight: Optional[dict]=None,
        show_plots: bool=True) -> History:
    """
    Train the keras model based on the parameters.

    Parameters
    ----------
    model : keras.models
        The model to train.
    X_train : np.ndarray
        The training data.
    y_train : np.ndarray
        The training labels.
    X_val : np.ndarray
        The validation data.
    y_val : np.ndarray
        The validation labels.
    X_test : np.ndarray
        The test data.
    y_test : np.ndarray
        The test labels.
    epochs : int
        The number of epochs.
    batch_size : int
        The batch size.
    optimizer : keras.optimizers
        The optimizer.
    loss : str
        The loss function.
    metrics : list
        The metrics.
    class_weight : dict, optional
        The class weights, by default None
    
    Returns
    -------
    keras.callbacks.History
        The history of the training.
    """
    # create the callback
    test_callback = TestModelCallback(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        show_plots=show_plots)

    # compile the model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)
    
    # fit the model
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=[test_callback])
    
    return history

def train_and_test_model(
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val : np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int,
        batch_size: int,
        optimizer,
        loss: str,
        metrics: list,
        class_weight: Optional[dict]=None,
        clear_learning: bool = False,
        show_plots: bool = True) -> None:
    """
    Train and test the model.

    Parameters
    ----------
    model : keras.models
        The model to train and test.

    X_train : np.ndarray
        The training data.
    y_train : np.ndarray
        The training labels.

    X_val : np.ndarray
        The validation data.
    y_val : np.ndarray
        The validation labels.

    X_test : np.ndarray
        The test data.
    y_test : np.ndarray
        The test labels.

    epochs : int
        The number of epochs.
    batch_size : int
        The batch size.
    optimizer : keras.optimizers
        The optimizer.
    loss : str
        The loss function.
    metrics : list
        The metrics.
    class_weight : dict, optional
        The class weights, by default None

    Returns
    -------
    float
        The optimized threshold for the best F1 score.
    """
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        class_weight=class_weight,
        show_plots=show_plots)
    
    # clear the learning output if required
    if clear_learning:
        clear_output()

def log_params(dataset:dict,
               feature_list:List[str],
               random_state:int) ->None:
    """
    Log the parameters for the model to Mlflow.

    Parameters
    ----------
    dataset : dict
        The dataset.
    feature_list : List[str]
        The list of features.
    random_state : int
        The random state.

    Returns
    -------
    None
    """
    mlflow.log_param('feature_list', feature_list)
    mlflow.log_param('random_state', random_state)    

    mlflow.log_param('train_shape', dataset['train']['X'].shape)
    mlflow.log_param('val_shape', dataset['val']['X'].shape)
    mlflow.log_param('test_shape', dataset['test']['X'].shape)
