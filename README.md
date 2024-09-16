# CSC311 Machine Learning Challenge Report


This project is a team-based machine learning challenge focused on building a classifier for predicting AI-generated image references from student survey responses. 

## Objective

- Develop a machine learning model to classify student responses about AI-generated images.
- The final model will be evaluated on an unseen test set.
- Ensure that your model handles bias-variance trade-offs effectively.

## Data

The dataset includes several subjective and objective survey responses. We omitted highly subjective features (e.g., `q_scary`, `q_dream`, `q_desktop`, `q_sell`) due to bias in personal opinions. Key features like `q_better`, `q_temperature`, `q_remind`, and `q_quote` were retained for model training as they provided a more objective representation of the environment depicted in each image.

- **q_story** was particularly valuable due to its descriptive nature. Stories were processed using a bag-of-words approach, with each word in the dataset forming the vocabulary for training.
- **q_temperature** was modeled using k-Nearest Neighbors (kNN), while other features were modeled using Naive Bayes.

## Model
We explored several models, including Naive Bayes, kNN, clustering, feature mapping, and decision trees:

- **Naive Bayes** was used for `q_quote`, `q_better`, `q_remind`, and `q_story` due to the independence assumption of the features. The bag-of-words method helped classify text-based features.
- **kNN** was applied to `q_temperature`, as it was better suited to numerical data. However, kNN underperformed and was eventually excluded from the final model.
- **Feature mapping** was considered but dropped due to incompatibility between Naive Bayes’ probabilistic outputs and kNN’s categorical outputs.
- **Decision Trees** were ruled out due to inefficiency without external libraries (e.g., `sklearn`) and difficulty in handling outliers.

## Model Choice and Hyperparameters
The dataset was split 70/30 into training (400 data points) and validation (172 data points). Validation accuracy was used to assess model performance.

- **Naive Bayes** performed well with hyperparameters `alpha=2` and `beta=2`, achieving a validation accuracy of 92%.
- **kNN** showed consistently low validation accuracy for any value of `k` (1-200), so it was not included in the final predictions.

## Prediction
We expect the model to perform well on the test set, with an estimated test accuracy of around 80%. The high accuracy is attributed to the strong correlation between the chosen features and the images, particularly the detailed text descriptions in `q_story`.

## Workload Distribution
- **Zhizhang Ma**: Programming, theoretical analysis.
- **Muhammad Haris Idrees**: Group organization, model programming, hyperparameter discussion, report contribution.
- **Samantha Skeels**: Model choices, hyperparameter discussion, report writing.