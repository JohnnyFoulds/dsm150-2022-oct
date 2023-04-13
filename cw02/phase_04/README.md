# Phase 04

## Kaggle Notebook Review

source: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/code?competitionId=45533&sortBy=dateRun

### Random Forest Baseline - [0.664] :: 001

_by [CHRIS DEOTTE](https://www.kaggle.com/code/cdeotte/random-forest-baseline-0-664)_

#### Feature Engineering

For the categorical features: `['event_name', 'name','fqid', 'room_fqid', 'text_fqid']` the following features are created only the unique count (`nunique`) is used.

_Note: We can consider adding the most occurring feature (`mode`)._

For the numerical features: `['elapsed_time','level','page','room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration']` the `mean` and `std` are used.

_Note: Notice in particular the `page` columns which we completely ignored. _

#### Training

- KFold Cross Validation is used (although I don't see how he actually uses the separate model created in each fold for each question).
- A separate model is trained for each question.
- Each question is predicted using its own model, but to optimize the f1 score, a single threshold is applied across all question correct probabilities.
