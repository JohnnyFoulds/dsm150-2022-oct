"""
This module implements a base class to use for prediction.
"""

from abc import ABC, abstractmethod
import logging
from typing import List
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# create an abstract base class
class PredictionBase(ABC):
    """
    The abstract base class for all prediction classes.
    """

    def __init__(self):
        """
        Initialize the base class
        """

    @abstractmethod
    def feature_engineering(self, data:pd.DataFrame, labels:pd.DataFrame) -> pd.DataFrame:
        """
        This method is used to perform feature engineering on the data.
        """

    @abstractmethod
    def predict_question(self, feature_set:List[np.ndarray], question_num:int) -> int:
        """
        This method is used to predict the target variable.
        """
    
    @abstractmethod
    def create_feature_dataset(self,
                               features:pd.DataFrame,
                               labels:pd.DataFrame) -> List[np.ndarray]:
        """
        This method is used to create a feature dataset.
        """

    def predict(self, data:pd.DataFrame, labels:pd.DataFrame) -> pd.DataFrame:
        """
        This method is used to predict the target variable.
        """
        # create the data frame to hold the predictions
        df_predictions = labels.copy()
        label_predictions = []

        # perform the feature engineering
        df_features = self.feature_engineering(data, labels)

        # process each row in the labels data
        logging.info('Predicting the target variable')
        for index, row in tqdm(labels.iterrows(), total=labels.shape[0]):
            df_session = df_features[df_features.session_id == row.session_id]
            df_question = df_session[df_session.level_group == row.level_group]
                              
            # create the feature dataset
            question_features = self.create_feature_dataset(
                features=df_question,
                labels=labels.loc[[index]])

            # predict the target variable
            label = self.predict_question(question_features, row.question_num)
            label_predictions.append(label)

        # assign the label to the predictions data frame
        df_predictions['correct'] = label_predictions

        return df_predictions

class Baseline(PredictionBase):
    """
    This baseline class predicts the target variable using a simple 
    answer array.
    """

    def __init__(self):
        """
        Initialize the baseline class
        """
        self.answer_key = [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1]

    def feature_engineering(self, data:pd.DataFrame, labels:pd.DataFrame) -> pd.DataFrame:
        """
        This method is used to perform feature engineering on the data.
        """
        return data

    def create_feature_dataset(self,
                               features:pd.DataFrame,
                               labels:pd.DataFrame) -> List[np.ndarray]:
        """
        This method is used to create a feature dataset.
        """
        return [features.values]

    def predict_question(self, feature_set:List[np.ndarray], question_num:int) -> int:
        """
        This method is used to predict the target variable.
        """
        return self.answer_key[question_num - 1]

