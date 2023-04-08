"""
This module implements a base class to use for prediction.
"""

from abc import ABC, abstractmethod
from typing import List
import pandas as pd

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
    def feature_engineering(self, data:pd.DataFrame, labels:pd.DataFrame) -> List[pd.DataFrame]:
        """
        This method is used to perform feature engineering on the data.
        """

    @abstractmethod
    def predict_question(self, feature_set:List[pd.DataFrame], question_num:int) -> int:
        """
        This method is used to predict the target variable.
        """

    def predict(self, data:pd.DataFrame, labels:pd.DataFrame) -> pd.DataFrame:
        """
        This method is used to predict the target variable.
        """
        # create thed data frame to hold the predictions
        df_predictions = labels.copy()
        label_predictions = []

        # perform the feature engineering
        feature_set = self.feature_engineering(data, labels)

        # process each row in the labels data
        for index, row in labels.iterrows():
            session_features = [df_features[df_features.session_id == row.session_id]
                                for df_features in feature_set]
                                
            question_features = [df_features[df_features.level_group == row.level_group]
                                 for df_features in session_features]

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

    def feature_engineering(self, data:pd.DataFrame, labels:pd.DataFrame) -> List[pd.DataFrame]:
        """
        This method is used to perform feature engineering on the data.
        """
        return [data]

    def predict_question(self, feature_set:List[pd.DataFrame], question_num:int) -> int:
        """
        This method is used to predict the target variable.
        """
        return self.answer_key[question_num - 1]

