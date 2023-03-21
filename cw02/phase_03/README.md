# Phase 3

Hours after submitting my assignment (`phase_02`), the landscape of the competition has change drastically in terms of leaked data and additional information.

1. [Leaked Competition Data](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/396202) - The private test set has been leaked and the organizers have now included it in the training data available for download which double the amounts of data available for training. 

2. [Raw Game Data](https://fielddaylab.wisc.edu/opengamedata/) - The raw game data is now available for download. What is particularly interesting to to look at some of the features. For example you examine the [`Sessions`](https://opengamedata.fielddaylab.wisc.edu/data/JOWILDER/JOWILDER_20220901_to_20220930_6228b2e_session-features.zip) dataset there is an interesting list of features that might be helpful for doing our own feature engineering.

3. [Feature Engineering](https://opengamedata.fielddaylab.wisc.edu/data/JOWILDER/readme.md) - There is several things here from the github repository of interest, especially the [code](https://github.com/opengamedata/opengamedata-core/tree/master/extractors) they used to generate the [features](https://github.com/opengamedata/opengamedata-core/tree/master/games/JOWILDER/features).

### API Changes
There has also been changes to the API which might have been the reason I have also seen exceptions when trying to do a submission. It is worth trying again with this update:

_"Please note that the order of the dataframes has changed - iter_test will now yield (test, sample_submission) rather than (sample_submission, test). This was in response to a bug report where the API was throwing exceptions for some people."_

### Different Game Events

It also seems like that it might be possible that not all users are playing the same game events.

- https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/396068
- https://www.kaggle.com/code/steubk/meetings-are-boring-the-notebook
- https://www.kaggle.com/datasets/steubk/meetings-are-boring

In the training data not all users are playing the same game events because there are four different scripts that are randomly chosen when the game starts.

```
Script Versions

Starting in v7, multiple scripts were added to the game for AB tests on snark and humor. The game will randomly choose between 4 different data files: data_dry.js, data_nohumor.js, data_nosnark.js, and the original data.js. They are each built from their respective data folder in assets. The type of script used is only logged once, in startgame.

Index Name Description
0 dry no humor or snark
1 nohumor no humor (includes snark)
2 nosnark no snark (includes humor). No snark can also be thought of as "obedient"
3 normal base script (includes snark and humor)
```

If you want to play a specific game type you must use these custom links:

normal: https://jowilder-master.netlify.app/?script_type=original
dry: https://jowilder-master.netlify.app/?script_type=dry
nohumor: https://jowilder-master.netlify.app/?script_type=nohumor
nosnark: https://jowilder-master.netlify.app/?script_type=nosnark




