import logging
from typing import Optional, Dict

import pandas as pd
import matplotlib.pyplot as plt

COUNT_COLUMNS = {
    'event_name': { 
        'total': {'min': 91.0, 'max': 924.5 },
        'unique': {'min': 7.0, 'max': 11.0 }
    },
    'name': { 
        'total': {'min': 91.0, 'max': 924.5 },
        'unique': {'min': 3.0, 'max': 6.0 }
    },
    'fqid': { 
        'total': {'min': 64.0, 'max': 683.0 },
        'unique': {'min': 18.0, 'max': 77.0 }
    },
    'room_fqid': { 
        'total': {'min': 91.0, 'max': 924.5 },
        'unique': {'min': 6.0, 'max': 17.0 }
    },
    'text_fqid': { 
        'total': {'min': 45.0, 'max': 359.5 },
        'unique': {'min': 8.0, 'max': 48.0 }
    },
}

def create_initial_features(X:pd.DataFrame,
                            y:pd.DataFrame) -> pd.DataFrame:
    """
    Creates the initial dataset to which additional features will be added.

    Parameters
    ----------
    X : pd.DataFrame
        The main dataset.
    y : pd.DataFrame
        The label dataset.

    Returns
    -------
    pd.DataFrame
        The initial feature dataset.
    """
    df_features =  y \
        .groupby(['session_id', 'level_group']) \
        .agg({'correct': ['count']}) \
        .reset_index() \
        .droplevel(1, axis=1) \
        .drop(columns=['correct']) \
        .sort_values(['session_id', 'level_group']) \
        
    # set the session_id to be an integer
    df_features['session_id'] = df_features['session_id'].astype(int)
        
    return df_features

def add_numeric_column_features(features:pd.DataFrame,
                                X:pd.DataFrame,
                                column:str,
                                min_values:Optional[dict]=None,
                                max_values:Optional[dict]=None) -> pd.DataFrame:
    """
    Add the maximum elapsed time feature to the features dataset.

    Parameters
    ----------
    features : pd.DataFrame
        The features dataset.
    X : pd.DataFrame
        The main dataset.
    column : str
        The name of the numeric column to add to the features for.

    Returns
    -------
    None
    """
    # Define a function to calculate mode
    def mode(series):
        return series.mode().iat[0]

    # calculate the maximum, minimum and mean for the column
    df_result = X \
        .groupby(['session_id', 'level_group']) \
        .agg({column: ['sum', 'max', 'min', 'mean', mode]}) \
        .reset_index()
    
    # flatten the multi-index columns
    df_result.columns = \
        ['_'.join(col).rstrip('_') for col in df_result.columns.values]

    # normalize the values
    if min_values is None or max_values is None:
        logging.warning('Not normalizing the values, min_value and max_values are not set.')
    else:
        metric_list = ['sum', 'max', 'min', 'mean', 'mode']
        for metric in metric_list:
            current_column = f'{column}_{metric}'
            df_result[current_column] = \
                (df_result[current_column] - min_values[metric]) / \
                (max_values[metric] - min_values[metric])       

    # join the features to the result   
    df_result = features.set_index(['session_id', 'level_group']) \
        .join(df_result.set_index(['session_id', 'level_group']), how='left') \
        .reset_index()
    
    return df_result

def add_elapsed_time_features(features:pd.DataFrame,
                              X: pd.DataFrame) -> pd.DataFrame:
    return add_numeric_column_features(
        features=features,
        X=X,
        column='elapsed_time',
        min_values={
            'sum': 61395.0,
            'max':  990.0,
            'min':  0.0,
            'mean': 526.447,
            'mode': 0.0},
        max_values={
            'sum':  9990648000,
            'max':  3691298.0,
            'min':  3691298.0,
            'mean': 3691298.0,
            'mode': 3691298.0})    

def plot_numeric_features(df_features:pd.DataFrame,
                          colum:str) -> None:
    """
    Plot the numeric features for a column.
    """
    metric_list = ['sum', 'max', 'min', 'mean', 'mode']
    column_list = [f'{colum}_{metric}' for metric in metric_list]

    # plot the features
    df_features[column_list].plot(
        kind='box',
        subplots=True,
        layout=(2, 3),
        figsize=(15, 10))
    
    plt.show()

def add_count_total_features(features:pd.DataFrame,
                             X:pd.DataFrame,
                             columns:dict) -> pd.DataFrame:
    """
    Add the total count for the categorical columns to the features dataset.

    Parameters
    ----------
    features : pd.DataFrame
        The features dataset.
    X : pd.DataFrame
        The main dataset.
    columns : dict
        The columns to add to the features dataset as a dictionary 
        of column name and min & max value.

    Returns
    -------
    pd.DataFrame
        The features dataset with the total count features added.
    """
    df_count_total = X \
        .groupby(['session_id', 'level_group']) \
        .agg({col: 'count' for col in columns.keys()}) \
        .reset_index()
    
    # normalize the counts
    for col, min_max in columns.items():
        # clip the values
        df_count_total[col] = df_count_total[col].clip(
            min_max['total']['min'],
            min_max['total']['max'])

        # normalize the values
        df_count_total[col] = \
            (df_count_total[col] - min_max['total']['min']) / \
            (min_max['total']['max'] - min_max['total']['min'])

    # join the features to the result
    df_result = features.set_index(['session_id', 'level_group']) \
        .join(
            df_count_total.set_index(['session_id', 'level_group']) \
                .add_prefix('count_total_'),
            how='left') \
        .reset_index()
    
    return df_result

def add_total_features(features:pd.DataFrame,
                               X:pd.DataFrame) -> pd.DataFrame:
    """
    Add the total elapsed time features to the dataset.

    Parameters
    ----------
    features : pd.DataFrame
        The features dataset.
    X : pd.DataFrame
        The main dataset.

    Returns
    -------
    pd.DataFrame
        The features dataset with the total elapsed time features added.
    """
    return add_count_total_features(
        features=features,
        X=X,
        columns=COUNT_COLUMNS)

def add_count_unique_features(features:pd.DataFrame,
                              X:pd.DataFrame,
                              columns:dict) -> pd.DataFrame:
    """
    Add the unique count for the categorical columns to the features dataset.

    Parameters
    ----------
    features : pd.DataFrame
        The features dataset.
    X : pd.DataFrame
        The main dataset.
    columns : dict
        The columns to add to the features dataset as a dictionary 
        of column name and min & max value.

    Returns
    -------
    pd.DataFrame
        The features dataset with the total count features added.
    """
    df_count_total = X \
        .groupby(['session_id', 'level_group']) \
        .agg({col: 'nunique' for col in columns.keys()}) \
        .reset_index()
    
    # normalize the counts
    for col, min_max in columns.items():
        # clip the values
        df_count_total[col] = df_count_total[col].clip(
            min_max['unique']['min'],
            min_max['unique']['max'])

        # normalize the values
        df_count_total[col] = (df_count_total[col] - min_max['unique']['min']) / \
            (min_max['unique']['max'] - min_max['unique']['min'])

    # get the columns as a feature vector
    count_total_feature = df_count_total[columns.keys()].to_numpy()
    df_count_total['count_unique_feature'] = pd.Series(count_total_feature.tolist())

    # drop the original columns
    df_count_total.drop(columns=columns.keys(), inplace=True)

    # add the feature to the features dataset
    df_result = features.set_index(['session_id', 'level_group']) \
        .join(df_count_total.set_index(['session_id', 'level_group']), how='left') \
        .reset_index()
    
    return df_result