import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import logging
from typing import Optional, Tuple, Iterable

from sklearn.model_selection import train_test_split

def select_sessions(
        y: pd.DataFrame,
        random_state: int=1337,
        test_size: float=0.2,
        train_size:float=0.6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select samples from the dataset for training, validation and testing.
    The test set is selected first, then the training set is selected from the 
    remaining sessions. And finally the validation set is selected from the
    remaining sessions.

    Parameters
    ----------
    y : pd.DataFrame
        The label dataset.
    random_state : int
        The random state to use.
    test_size : float
        The ratio of the sample to use for testing.
    train_size : float
        The ratio of the sample to use for training.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The selected session ids, the main dataset and the label dataset.
    """
    # select all the unique session ids
    all_session_ids = y['session_id'].unique()

    # set the random seed
    np.random.seed(random_state)

    # shuffle the session ids
    np.random.shuffle(all_session_ids)

    # select the session ids for the test set
    test, remainder = train_test_split(all_session_ids, test_size=1-test_size)

    # split the dataset into train and validation sets
    train, val = train_test_split(remainder, test_size=1-train_size)

    # print the number of sessions in each set
    print(f'Train: {len(train)}')
    print(f'Validation: {len(val)}')
    print(f'Test: {len(test)}')

    return train, val, test

def create_feature_dataset(df_features:pd.DataFrame,
                           df_source_labels:pd.DataFrame,
                           session_list: list,
                           feature_list:list,
                           level_group:Optional[str]=None,
                           include_question:bool=True,
                           expand_question:bool=False) -> np.ndarray:
    """
    Creates the feature dataset for the given level group and session list.
    If the level group is not specified it will create the dataset for all level groups.

    Parameters
    ----------
    df_features : pd.DataFrame
        The dataset of prepared features (by session_id and level_group).
    df_source_labels : pd.DataFrame
        The dataset containing the training labels (y_True).
    session_list : list
        The list of session ids to create the dataset for.
    level_group : str, optional
        The level group to create the dataset for, by default None
    feature_list : list
        The list of features to include in the dataset.
    include_question : bool, optional
        Whether to include the question number in the dataset as the first set of
        columns, by default True
    expand_question : bool, optional
        Whether to expand the question number into a one-hot vector to each item in the 
        case of a multi-dimensional feature, by default False

    Returns
    -------
    np.array
        The feature dataset.
    """
    # get the features and labels for the given level group
    if level_group is None:
        logging.info('Creating the dataset for all level groups')
        df_features_group = df_features.query('session_id in @session_list')
        df_labels_group = df_source_labels.query('session_id in @session_list')
    else:
        logging.info('Creating the dataset for level group: %s', level_group)
        df_features_group = df_features.query('level_group == @level_group and session_id in @session_list')
        df_labels_group = df_source_labels.query('level_group == @level_group and session_id in @session_list')

    # sort the df_labels_group
    df_labels_group = df_labels_group.sort_values(['session_id', 'question_num'])

    feature_dataset = []

    # get the features for each row in the level group labels dataset
    current_session_id = None
    df_session_features:pd.DataFrame = None

    for index, row in tqdm(df_labels_group.iterrows(), total=df_labels_group.shape[0]):        
        session_id = int(row['session_id'])
        session_level_group = row['level_group']
        question_num = int(row['question_num'])

        # get the features for the session
        if session_id != current_session_id:
            current_session_id = session_id
            df_session_features = df_features_group.query('session_id == @session_id')

        # get the level group features
        df_level_group_features = df_session_features.query('level_group == @session_level_group')

        # check if the session has features
        if df_level_group_features.shape[0] == 0:
            raise Exception(f'No features for session {session_id}, level group {session_level_group}!')
                            
        # get the features for the row
        row_features = []

        # get the question number one-hot encoded
        question_num_one_hot = np.zeros(18, dtype=np.int8)
        question_num_one_hot[question_num-1] = 1

        if include_question:
            row_features.extend(question_num_one_hot)

        for feature in feature_list:
            feature_value = df_level_group_features[feature].values[0]

            # check if the feature value is iterable
            if isinstance(feature_value, Iterable):
                if expand_question:
                    # reshape the question array to match the feature array shape
                    question_reshaped = np.tile(
                        question_num_one_hot, 
                        (feature_value.shape[0], 1))
                    
                    # add the question columns to the feature array
                    feature_value = np.hstack((question_reshaped, feature_value))

                row_features.extend(feature_value)
            else:
                row_features.append(feature_value)

        # add the row features to the output dataset
        feature_dataset.append(row_features)

    return np.array(feature_dataset, dtype=np.float32)

def create_label_dataset(session_list: list,
                          df_source_labels:pd.DataFrame) -> np.ndarray:
    """
    Create the y_true values for the given session list.

    Parameters
    ----------
    session_list : list
        The list of session ids to create the dataset for.
    df_source_labels : pd.DataFrame
        The dataset containing the training labels (y_True).

    Returns
    -------
    np.array
        The y_true dataset.
    """
    # get the relevant sessions
    answers = df_source_labels \
        .query('session_id in @session_list') \
        .sort_values(by=['session_id', 'question_num']) \
        .correct \
        .values
    
    return np.array(answers, dtype=np.int8)

def get_feature_dataset(features:pd.DataFrame,
                        y:pd.DataFrame,
                        feature_list:list,
                        train: list,
                        val: list,
                        test: list,
                        include_question:bool=True,
                        expand_question:bool=False) -> dict:
    """
    Create a dictionary containing the features for the train,
    validation and test datasets.

    Parameters
    ----------
    features : pd.DataFrame
        The dataset of prepared features (by session_id and level_group).
    y : pd.DataFrame
        The dataset containing the training labels (y_True).
    feature_list : list
        The list of features to include in the dataset.
    train : list
        The list of session ids for the training dataset.
    val : list
        The list of session ids for the validation dataset.
    test : list
        The list of session ids for the test dataset.
    include_question : bool, optional
        Whether to include the question number in the dataset as the 
        first set of columns, by default True
    expand_question : bool, optional
        Whether to expand the question number into a one-hot vector to
        each item in the case of a multi-dimensional feature,
        by default False

    Returns
    -------
    dict
        The dictionary containing the feature datasets for the train,
        validation and test
    """
    feature_dataset = {}
    for session_list, name in [(train, 'train'), (val, 'val'), (test, 'test')]:
        logging.info('-- Creating the %s dataset', name)
        feature_dataset[name] = {}

        # get the X values
        feature_dataset[name]['X'] = create_feature_dataset(
            df_features=features,
            df_source_labels=y,
            session_list=session_list,
            feature_list=feature_list,
            include_question=include_question,
            expand_question=expand_question)
        
        # get the y values
        feature_dataset[name]['y'] = create_label_dataset(
            session_list=session_list,
            df_source_labels=y)
        
    return feature_dataset