# Decision-Tree-for-Spotify-Data
Implementation of a Decision Tree classifier using scikit-learn for use on the Spotify dataset to determine whether the user would like the song or not.

# Feature Selection
These are all the features available within the dataset:
1. acousticness
1. danceability
1. duration_ms
1. energy
1. instrumentalness
1. key
1. liveness
1. loudness
1. mode
1. speechiness
1. tempo
1. time_signature
1. valence
1. target
1. song_title
1. artist

Since the maximum number of times a singular artist is included is 16, which is less than 0.8% of the entire dataset, the artist column would probably not be a good feature to split on for a binary classification task. Additionally, since there are 1956 unique song titles in this dataset, the song_title column would not be a good feature to split on for this task either. Thus, these 2 columns will be dropped.

To perform feature selection, I used mutual information as I thought it would be a good way to assess which features are most important and which ones are least important.Additionally, I thought it would be appropriate to use this metric as we are fitting decision trees to this data. Based on a threshold of mutual information = 0.01, The most important are loudness, duration_ms, danceability, instrumentalness, acousticness, energy, tempo, and speechiness while the least important features are time_signature, valence, mode, liveness, and key.

![image](https://github.com/amaank123456/Decision-Tree-for-Spotify-Data/assets/149258362/70c26a89-543c-4e35-b1bc-335770597d85)

# Model Training

A 5-fold cross-validation and grid search was used to tune the hyperparameters of the Decision Tree classifier (used criterion, max_depth, class_weight, min_samples_split, min_samples_leaf, and max_features).

The best-hyperparameter values were:

| Hyperparameter       | Value        |
|----------------------|--------------|
| class_weight         | balanced     |
| criterion            | entropy      |
| max_depth            | 5            |
| max_features         | log2         |
| min_samples_leaf     | 2            |
| min_samples_split    | 2            |



# Evaluating the Model


After obtaining the best-hyperparameter values, the model was retrained on all the training data, and the metrics for the test results are listed below:

| Metric   | Value |
|----------|-------|
| Precision| 0.74  |
| Recall   | 0.67  |
| Accuracy | 0.70  |
| F1-Score | 0.70  |

Here is the confusion matrix for the test predictions:

![image](https://github.com/amaank123456/Decision-Tree-for-Spotify-Data/assets/149258362/71c8dc08-a931-4b96-9b0b-ccf90fde61bb)

# Decision Tree Visualization

The Decision Tree can be visualized here: [Decision Tree Visualization](https://github.com/amaank123456/Decision-Tree-for-Spotify-Data/blob/main/decision_tree.pdf)



