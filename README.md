# Mushroom Classification Project

![](UTA-DataScience-Logo.png)

## Mushroom Classification Project

This project focuses on classifying mushrooms as either edible or poisonous using the Mushroom Classification Dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification). Using machine learning, the goal was to develop an accurate model to classify mushrooms based on their features.

## Overview

The dataset contains 8,124 rows and 22 categorical features describing mushrooms, such as cap shape, color, and odor. The target column (`class`) specifies whether a mushroom is edible (`e`) or poisonous (`p`). The dataset is balanced, making it well-suited for classification tasks:

![image](https://github.com/user-attachments/assets/e3491a74-2994-4970-b0c7-1a82df902d59)


To achieve 100% classification accuracy, Random Forest was selected as the final model after comparing it with Logistic Regression, K-Nearest Neighbors, and Support Vector Machines. This result was achieved by combining detailed data exploration, preprocessing, feature visualization, and model evaluation.

## Summary of Work

### Data Exploration

Key insights from exploration:
- The `veil-type` column had only one unique value and was removed.
- The `stalk-root` column contained missing values (`?`), which were replaced with "unknown."

The class distribution was nearly balanced:

![image](https://github.com/user-attachments/assets/f8825499-6ed6-4e06-8bc7-d8a1ff673625)

![](visualizations/class_distribution.png)

### Data Cleaning and Preprocessing

Data preparation steps:
- One-hot encoding was applied to all categorical features except the target.
- The target column (`class`) was encoded as `0` (edible) and `1` (poisonous).

### Data Visualization

Visualizations helped identify predictive features:

- **Feature Importance:** A Random Forest model highlighted the top 15 most important features, with `odor` and `gill-size` being the most predictive.

![image](https://github.com/user-attachments/assets/e3491a74-2994-4970-b0c7-1a82df902d59)

![image](https://github.com/user-attachments/assets/f8825499-6ed6-4e06-8bc7-d8a1ff673625)

- **Odor Feature:** The `odor` feature strongly distinguished between edible and poisonous mushrooms, showcasing a clear predictive pattern.

![image](https://github.com/user-attachments/assets/bd7b8984-0ff0-4c27-bb04-14d6111baf24)

![image](https://github.com/user-attachments/assets/bd7b8984-0ff0-4c27-bb04-14d6111baf24)

### Problem Formulation

The task was framed as a binary classification problem:
- **Input:** Preprocessed features after one-hot encoding.
- **Output:** Binary classification (`0` for edible, `1` for poisonous).

Models tested:
- Logistic Regression
- Random Forest
- K-Nearest Neighbors
- Support Vector Machines

### Model Training and Comparison

Each model was trained on a 70% training set and evaluated on a 15% validation set. Results:

| Model                  | Validation Accuracy | Test Accuracy | Notes                                   |
|------------------------|---------------------|---------------|-----------------------------------------|
| Logistic Regression    | 99.75%             | 99.75%        | Simple and interpretable.               |
| Random Forest          | 100%               | 100%          | Robust and provided feature importance. |
| K-Nearest Neighbors    | 100%               | 100%          | Computationally expensive.              |
| Support Vector Machines| 100%               | 100%          | High accuracy but less interpretable.   |

Random Forest was chosen for its perfect accuracy and interpretability through feature importance.

### Test Set Evaluation

Random Forest achieved 100% accuracy on the test set, confirming its reliability and generalizability. The preprocessing and modeling steps ensured a robust solution.

### Submission

Predictions from the Random Forest model were saved in `submission.csv`. A preview:

| Index | Prediction |
|-------|------------|
| 0     | 1          |
| 1     | 0          |
| 2     | 1          |
| ...   | ...        |

## Conclusions

The project successfully classified mushrooms with perfect accuracy using Random Forest. The key elements that ensured this success were:
- Thorough cleaning and preprocessing.
- Visualization to identify important features.
- Careful model evaluation and selection.

## Future Work

Potential areas for further exploration:
- Hyperparameter tuning for models like Random Forest and Support Vector Machines.
- Deeper analysis of feature importance to better interpret results.
- Testing the methodology on similar classification datasets.

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

