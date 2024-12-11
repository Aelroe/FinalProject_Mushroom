# Mushroom Classification Project

![](UTA-DataScience-Logo.png)

## Mushroom Classification Project

This project focuses on classifying mushrooms as either edible or poisonous using the Mushroom Classification Dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification). By leveraging machine learning techniques, the goal was to create a model that accurately predicts mushroom edibility based on their features.

## Overview

The dataset contains 8,124 rows and 22 categorical features that describe various physical attributes of mushrooms, such as cap shape, color, and odor. The target column (`class`) indicates whether a mushroom is edible (`e`) or poisonous (`p`).

To achieve this, I followed a systematic approach:
1. Explored the dataset to identify potential issues.
2. Cleaned the data to handle missing values and irrelevant features.
3. Visualized feature distributions to understand relationships.
4. One-hot encoded categorical features to prepare the data for machine learning.
5. Trained and compared multiple models to find the best-performing classifier.

## Summary of Work

### Data Exploration

The initial exploration revealed:
- The `veil-type` column had only one unique value and was dropped.
- The `stalk-root` column contained missing values (`?`), which were replaced with "unknown."

A brief visualization of the target class distribution:

![](visualizations/class_distribution.png)

### Data Cleaning and Preprocessing

To prepare the data:
- One-hot encoding was applied to all categorical features except the target.
- The target column (`class`) was encoded as `0` (edible) and `1` (poisonous).

### Data Visualization

Feature distributions were compared between edible and poisonous classes. For example, the `odor` feature showed clear distinctions:

![](visualizations/odor_feature.png)

Such visualizations guided model selection by highlighting features with strong predictive potential.

### Problem Formulation

The task was treated as a classification problem:
- **Input:** Preprocessed features after one-hot encoding.
- **Output:** Binary classification (`0` for edible, `1` for poisonous).
- **Models Tested:** Logistic Regression, Random Forest, K-Nearest Neighbors, and Support Vector Machines.

### Model Training and Comparison

Several models were trained and evaluated using accuracy, precision, recall, and F1-scores. The results are summarized below:

| Model                  | Validation Accuracy | Test Accuracy | Notes                                   |
|------------------------|---------------------|---------------|-----------------------------------------|
| Logistic Regression    | 99.75%             | 99.75%        | Simple and interpretable.               |
| Random Forest          | 100%               | 100%          | Robust and provided feature importance. |
| K-Nearest Neighbors    | 100%               | 100%          | Computationally expensive.              |
| Support Vector Machines| 100%               | 100%          | High accuracy but less interpretable.   |

Random Forest was chosen as the final model for its perfect accuracy and feature importance insights.

### Test Set Evaluation

The Random Forest model achieved 100% accuracy on the test set, confirming its reliability and generalizability.

### Submission

The trained model's predictions were saved in a `submission.csv` file for evaluation. Here is a snippet of the predictions:

| Index | Prediction |
|-------|------------|
| 0     | 1          |
| 1     | 0          |
| 2     | 1          |
| ...   | ...        |

## Conclusions

The project successfully classified mushrooms with perfect accuracy using Random Forest. Key factors in achieving this result included thorough data preprocessing, effective feature encoding, and careful model evaluation.

## Future Work

- Experiment with additional models or hyperparameter tuning.
- Investigate feature importance to further interpret the model.
- Apply the methodology to similar datasets for broader validation.

## How to Reproduce Results

1. **Preprocessing:** Run the provided Jupyter Notebook (`Final_Project.ipynb`) to clean and preprocess the data.
2. **Training:** Train the models as outlined in the notebook.
3. **Evaluation:** Validate and test model performance using the provided code.
4. **Submission:** Use the `submission.csv` file for evaluation or sharing results.

## Files in Repository

- `Final_Project.ipynb`: Contains all steps, from data cleaning to model evaluation.
- `submission.csv`: Final predictions for the test set.
- `README.md`: Project description and documentation.

## Software Setup

- **Required Packages:**
  - pandas
  - scikit-learn
  - matplotlib
- **Installation:**
  ```bash
  pip install pandas scikit-learn matplotlib
  ```

## Citations

- Mushroom Classification Dataset: [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification).
- Scikit-learn documentation: https://scikit-learn.org/stable/index.html.

