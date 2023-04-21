"""
This module implements a base class to use for prediction.
"""

from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from competition import feature_engineering as fe
from competition import model_data as md

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


class HeatmapPredictor(Baseline):
    """
    Use the best model for question 5 to predict the correct labels.
    """

    def __init__(self, models:Dict[int, Dict[Any, float]]):
        # call the base class constructor
        super().__init__()

        # initialize the models collection
        self.models = {}
        self.thresholds = {}

        # assign the models
        self.models = models

    def feature_engineering(self, data:pd.DataFrame, labels:pd.DataFrame) -> pd.DataFrame:
        """
        This method is used to perform feature engineering on the data.
        """
        logging.info('Performing feature engineering:')
        # create the initial features
        df_features = fe.create_initial_features(data, labels)

        # add the elapsed time feature to the features dataset
        logging.info('\tAdding elapsed time features...')
        df_features = fe.add_elapsed_time_features(
            features=df_features,
            X=data)

        # add the total count features to the features dataset
        logging.info('\tAdding total count features...')
        df_features = fe.add_count_total_features(
            features=df_features,
            X=data)

        # add the unique count features to the features dataset
        logging.info('\tAdding unique count features...')
        df_features = fe.add_count_unique_features(
            features=df_features,
            X=data)

        # add the heatmap features to the features dataset
        logging.info('\tAdding heatmap features...')
        df_features = fe.add_screen_heatmap_feature(
            features=df_features,
            X=data,
            verbose=True)
        
        return df_features

    def create_feature_dataset(self,
                               features:pd.DataFrame,
                               labels:pd.DataFrame) -> List[np.ndarray]:
        """
        This method is used to create a feature dataset.
        """   
        # create the flat features dataset
        features_dataset = md.create_feature_dataset(
            df_features=features,
            df_source_labels=labels,
            session_list=labels.session_id.unique(),
            feature_list=['elapsed_time_sum', 'elapsed_time_max', 
                          'elapsed_time_min', 'elapsed_time_mean', 
                          'elapsed_time_mode'],
            include_question=True,
            expand_question=False,
            verbose=False)

        # create the heatmap features dataset
        #logging.info('Creating the heatmap features dataset...')
        heatmap_dataset = md.create_feature_dataset(
            df_features=features,
            df_source_labels=labels,
            session_list=labels.session_id.unique(),
            feature_list=['screen_heatmap_feature'],
            include_question=False,
            expand_question=False,
            verbose=False)

        return [heatmap_dataset, features_dataset]


    def predict_question(self, feature_set:List[np.ndarray], question_num:int) -> int:
        """
        Predict the correct answer for the given question.

        Parameters
        ----------
        feature_set : List[pd.DataFrame]
            The list of feature sets for the questions.
        question_num : int
            The question number to predict.

        Returns
        -------
        int
            The predicted answer for the question.
        """
        # if no model is defined for the question, use the base class
        model_data = self.models.get(question_num, None)
        if model_data is None:
            return super().predict_question(feature_set, question_num)

        # get the model and threshold
        model = model_data['model']
        threshold = model_data['threshold']

        # use the model for prediction
        y_pred_model = model.predict(feature_set, verbose=0)
        y_pred_model = (y_pred_model[:, 1] > threshold).astype(int)

        return y_pred_model[0]