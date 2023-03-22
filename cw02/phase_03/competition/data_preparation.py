import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Optional, Set, Tuple

def map_question_to_level_group(question_number) -> Optional[str]:
    """
    Maps the question number to the level group.

    Parameters
    ----------
    question_number : int
        The question number.

    Returns
    -------
    str
        The level group.
    """
    if question_number in [1, 2, 3]:
        return '0-4'
    elif question_number in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        return '5-12'
    elif question_number in [14, 15, 16, 17, 18]:
        return '13-22'
    else:
        return None
    
def find_problem_sessions(data : pd.DataFrame) -> Set[str]:
    """
    Finds the sessions that are duplicated on session_id and index. And
    Find sessions with reversed indexes.

    This idea is taken from the following Kaggle notebook:
    https://www.kaggle.com/code/abaojiang/eda-on-game-progress/notebook?scriptVersionId=120133716
    
    Parameters
    ----------
    data : pd.DataFrame
        The data to search.

    Returns
    -------
    List[str]
        The list of session ids that have a problem.
    """

    # find sessions duplicated on session_id and index
    sessions_with_duplicates = data.loc[
        data.duplicated(subset=["session_id", "index"], keep=False)] \
        ["session_id"].unique().tolist()


    # find sessions with reversed indexes
    sessions_with_reversed_index = []
    for sess_id, gp in data.groupby("session_id", observed=True):
        if not gp["index"].is_monotonic_increasing:
            sessions_with_reversed_index.append(sess_id)

    # via experimentation these sessions have been found to have time 
    # differences < -2000
    negative_time_diff_sessions = [
        '21030417085341900', '21070111080982292', 
        '21090108302064196', '21090409222921812']

    # combine the two lists into a single set
    return set(sessions_with_duplicates + 
               sessions_with_reversed_index + 
               negative_time_diff_sessions)

def prepare_label_dataset(data : pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the label dataset and add columns for the level group 
    and the question number.

    Parameters
    ----------
    data : pd.DataFrame
        The label dataset.

    Returns
    -------
    pd.DataFrame
        The prepared label dataset.
    """
    # add the columns to determine the level group
    df_labels = data \
        .rename(columns={'session_id': 'id'}) \
        .assign(session_id=lambda df: df['id'].str.split('_').str[0].astype(int)) \
        .assign(question_id=lambda df: df['id'].str.split('_').str[1]) \
        .assign(question_num=lambda df: df['question_id'].str[1:].astype(int)) \
        [['session_id', 'question_num', 'correct']]
    
    # add the level group column
    df_labels['level_group'] = df_labels['question_num'] \
        .apply(map_question_to_level_group) 
        
    return df_labels

def prepare_main_dataset(data : pd.DataFrame,
                         elapsed_time_min_clip:int=0,
                         elapsed_time_max_clip:int=3691298) -> pd.DataFrame:
    """
    Prepares the main dataset by removing duplicates and removing 
    columns that are not needed.

    Parameters
    ----------
    data : pd.DataFrame
        The main dataset.

    Returns
    -------
    pd.DataFrame
        The prepared main dataset.
    """
    empty_columns = ['fullscreen', 'hq', 'music', 'page', 'hover_duration']

    df_main = data \
        .drop_duplicates() \
        .reset_index(drop=True) \
        .drop(empty_columns, axis=1) \
        .drop('text', axis=1)
    
    # clip the elapsed time to remove outliers
    df_main['elapsed_time'] = df_main['elapsed_time'].clip(
        lower=elapsed_time_min_clip,
        upper=elapsed_time_max_clip)
    
    return df_main

def get_clipping_values(data:pd.DataFrame, column:str, boxplot:bool=True) -> Tuple[float, float]:
    """
    To remove outliers, gets the clipping values for the specified column.

    Parameters
    ----------
    data : pd.DataFrame
        The data to search.
    column : str
        The column to search.
    boxplot : bool, optional
        If True, box plots are show for comparison, by default True.

    Returns
    -------
    Tuple[float, float]
        The clipping values.
    """
    # get the minimum and maximum values
    min_value = data[column].min()
    max_value = data[column].max()

    # get the inter quartile range
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)    
    iq_range = q3 - q1

    # get the clipping values
    min_clip = np.max([min_value, (q1 - (iq_range * 1.5))])
    max_clip = q3 + (iq_range * 1.5)

    # show the box plot
    if boxplot:
        # get the cliped values
        data_clipped = data[column].values.clip(min_clip, max_clip)

        # create the box plot next to each other
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        data[column].plot.box(ax=ax1)
        pd.Series(data_clipped).plot.box(ax=ax2)

        # set the title
        plt.suptitle(f'Box plot for {column}')
        ax1.set_title('Original')
        ax2.set_title('Clipped')

        # show the plot
        plt.show()

    return min_clip, max_clip