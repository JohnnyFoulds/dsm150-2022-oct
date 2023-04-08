import logging
from typing import Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

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

HEATMAP_BINS = 10
HEATMAP_MAX = 17

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
                             columns:dict=COUNT_COLUMNS) -> pd.DataFrame:
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

def add_count_unique_features(features:pd.DataFrame,
                              X:pd.DataFrame,
                              columns:dict=COUNT_COLUMNS) -> pd.DataFrame:
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

    # join the features to the result
    df_result = features.set_index(['session_id', 'level_group']) \
        .join(
            df_count_total.set_index(['session_id', 'level_group']) \
                .add_prefix('count_unique_'),
            how='left') \
        .reset_index()

    return df_result

def create_level_screen_heatmap(df_session:pd.DataFrame,
                                level:int,
                                bins:int=64,
                                normalize:bool=True,
                                min_value:int=0,
                                max_value:int=376) -> np.ndarray:
    """
    Creates a heatmap for the screen of the given level.

    Parameters
    ----------
    df_session : pd.DataFrame
        The dataframe containing the events for a single session.
    level : int
        The level to create the heatmap for.
    bins : int, optional
        The number of bins to use for the heatmap, by default 50
    min_value : int, optional
        The minimum value to use for the heatmap, by default 0
    max_value : int, optional
        The maximum value to use for the heatmap, by default 376

    Returns
    -------
    np.ndarray
        The heatmap for the given level.
    """
    df_level = df_session \
        .query('level == @level') \
        [['screen_coor_x', 'screen_coor_y']] \
        .fillna(0)
    
    # return an empty heatmap if there are no events for the given level
    if df_level.shape[0] == 0:
        return np.zeros((bins, bins), dtype=np.uint8)

    # Create the 2D histogram
    heatmap, xedges, yedges = np.histogram2d(df_level.screen_coor_y, 
                                             df_level.screen_coor_x, 
                                             bins=bins)
    
    # return the heatmap if no normalization is required
    if not normalize:
        return heatmap
    
    # clip the heatmap to the given min and max values
    heatmap = np.clip(heatmap, min_value, max_value)
    
    # Normalize the heatmap
    normalized_heatmap = (heatmap - min_value) / (max_value - min_value)

    # Scale the heatmap to the range 0-255
    scaled_heatmap = (normalized_heatmap * 255).astype(np.uint8)

    return scaled_heatmap

def create_level_group_screen_heatmap(df_session:pd.DataFrame,
                                      level_group:str,
                                      bins:int=HEATMAP_BINS,
                                      min_value:int=0,
                                      max_value:int=HEATMAP_MAX) -> np.ndarray:
    """
    Creates heatmaps for the screen of the given level group.
    """
    heatmaps = []
    level_range = range(0, 23)

    # select only the levels in the given level group
    df_level_group = df_session.query('level_group == @level_group')

    # create the heatmaps
    for level in level_range:
        heatmap = create_level_screen_heatmap(
            df_session=df_level_group, 
            level=level, 
            bins=bins, 
            normalize=True,
            min_value=min_value, 
            max_value=max_value)

        # normalize the heatmap to values between 0 and 1
        heatmap = heatmap / 255

        heatmaps.append(heatmap)

    return np.array(heatmaps, dtype=np.float32)

def add_screen_heatmap_feature(features:pd.DataFrame,
                               X:pd.DataFrame,
                               bins:int=HEATMAP_BINS,
                               min_value:int=0,
                               max_value:int=HEATMAP_MAX,
                               verbose:bool=True) -> pd.DataFrame:
    """
    Adds the screen heatmap feature to the features dataset.

    Parameters
    ----------
    features : pd.DataFrame
        The features dataset.
    X : pd.DataFrame
        The source dataset.
    bins : int, optional
        The number of bins to use for the heatmap, by default 10
    min_value : int, optional
        The minimum value to use for the heatmap, by default 0
    max_value : int, optional
        The maximum value to use for the heatmap, by default 17
    
    Returns
    -------
    pd.DataFrame
        The features dataset with the screen heatmap feature added.
    """
    heatmaps_feature = []
    for session_id in tqdm(features['session_id'].unique(), disable=not verbose):
        df_session = X[X['session_id'] == session_id]
        df_session_features = features[features['session_id'] == session_id]

        # process each level group the session features
        for index, row in df_session_features.iterrows():
            level_group = row['level_group']

            group_heatmaps = create_level_group_screen_heatmap(
                df_session, level_group, bins, min_value, max_value)
            
            heatmaps_feature.append(group_heatmaps)

    # add the feature to the features dataset
    df_features = features.copy()
    df_features['screen_heatmap_feature'] = pd.Series(heatmaps_feature)

    return df_features