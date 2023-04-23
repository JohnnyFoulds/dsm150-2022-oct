# Phase 04

## Kaggle Notebook Review

source: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/code?competitionId=45533&sortBy=dateRun

### Random Forest Baseline - [0.664] :: 0100

_by [CHRIS DEOTTE](https://www.kaggle.com/code/cdeotte/random-forest-baseline-0-664)_

#### Feature Engineering

For the categorical features: `['event_name', 'name','fqid', 'room_fqid', 'text_fqid']` the following features are created only the unique count (`nunique`) is used.

_Note: We can consider adding the most occurring feature (`mode`)._

For the numerical features: `['elapsed_time','level','page','room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']` the `mean` and `std` are used.

_Note: Notice in particular the `page` columns which we completely ignored. _

#### Training

- KFold Cross Validation is used.
- A separate model is trained for each question.
- Each question is predicted using its own model, but to optimize the f1 score, a single threshold is applied across all question correct probabilities.

_Note: It is a bit subtle to see, but each fold is used to predict a subset of the validation set. In essence, for each question K=5 models are created(one in each fold), and for validation each subset of K is predicted using the corresponding model. So we could say this is already and ensemble model albeit with the singe RF algorithm._

> - This is an interesting technique and we can try to incorporate it into our training pipeline. But perhaps we use the top 5 models we get during hyperparameter tuning.
> - **Note** though that it like for submission he only uses the models from the last fold for prediction, so perhaps the complexity of adding the CV is not worth it; will need to investigate.

### LightGBM baseline with aggregated log data :: 0200

_by [DATAMANYO](https://www.kaggle.com/code/kimtaehun/lightgbm-baseline-with-aggregated-log-data)_

- `def summary(df)` is a pretty neat function, I can use this to get a quick overview of the data.

#### Feature Engineering

 - Added one-hot encoding of `event_name`.
 - For the categorical values `text` is used as opposed to `text_fqid` in `001`. This make sense as we know now there are slight variations in the game types being played.
 - Additional features to `001` is the `sum` that are created from the one-hot features: `['navigate_click','person_click','cutscene_click','object_click','map_hover','notification_click', 'map_click','observation_click','checkpoint','elapsed_time']`.