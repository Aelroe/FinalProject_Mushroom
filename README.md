# Mushroom Classification Project

![](UTA-DataScience-Logo.png)

## Mushroom Classification Project

This project focuses on classifying mushrooms as either edible or poisonous using the Mushroom Classification Dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification). By leveraging machine learning techniques, the goal was to create a model that accurately predicts mushroom edibility based on their features.

## Overview

The dataset contains 8,124 rows and 22 categorical features that describe various physical attributes of mushrooms, such as cap shape, color, and odor. The target column (`class`) indicates whether a mushroom is edible (`e`) or poisonous (`p`).

This project followed a systematic approach:
1. Explored the dataset to identify potential issues.
2. Cleaned the data to handle missing values and irrelevant features.
3. Visualized feature distributions to understand relationships and identify predictive features.
4. One-hot encoded categorical features to prepare the data for machine learning.
5. Trained and compared multiple models to find the best-performing classifier, selecting Random Forest for its perfect performance.

## Summary of Work

### Data Exploration

During the initial exploration, the following key insights emerged:
- The `veil-type` column had only one unique value across all samples and was dropped.
- The `stalk-root` column contained missing values (`?`), which were replaced with "unknown."

Visualization of the target class distribution revealed a near-equal split between edible (`e`) and poisonous (`p`) mushrooms:

![](visualizations/class_distribution.png)

### Data Cleaning and Preprocessing

The dataset was cleaned and transformed to ensure it was ready for modeling:
- One-hot encoding was applied to all categorical features except the target column.
- The target column (`class`) was encoded as `0` (edible) and `1` (poisonous).

### Data Visualization

Visualizations were critical in identifying features with strong predictive power. For instance:
- **Odor Feature:** Clear distinctions were observed in the distribution of odor between edible and poisonous mushrooms.

![](visualizations/odor_histogram.png)

- **Cap Shape Feature:** The cap shape distribution also showed variability across classes, aiding feature selection.

![](visualizations/cap_shape_histogram.png)

These visualizations guided model selection and helped ensure the data was properly understood before training.

### Problem Formulation

The classification task was defined as:
- **Input:** Preprocessed features after one-hot encoding.
- **Output:** Binary classification (`0` for edible, `1` for poisonous).

The models tested included:
- Logistic Regression
- Random Forest
- K-Nearest Neighbors
- Support Vector Machines

Metrics such as accuracy, precision, recall, and F1-score were used to evaluate model performance.

### Model Training and Comparison

Each model was trained on a 70% training set and evaluated on a 15% validation set. The results are summarized below:

| Model                  | Validation Accuracy | Test Accuracy | Notes                                   |
|------------------------|---------------------|---------------|-----------------------------------------|
| Logistic Regression    | 99.75%             | 99.75%        | Simple and interpretable.               |
| Random Forest          | 100%               | 100%          | Robust and provided feature importance. |
| K-Nearest Neighbors    | 100%               | 100%          | Computationally expensive.              |
| Support Vector Machines| 100%               | 100%          | High accuracy but less interpretable.   |

Random Forest was chosen as the final model for its perfect accuracy and ability to highlight feature importance.

### Test Set Evaluation

The final model, Random Forest, achieved 100% accuracy on the test set, confirming its reliability and generalizability. This highlights the effectiveness of the model and the preprocessing steps undertaken.

### Submission

The trained model's predictions were saved in a `submission.csv` file. Here is a preview of the predictions:

| Index | Prediction |
|-------|------------|
| 0     | 1          |
| 1     | 0          |
| 2     | 1          |
| ...   | ...        |

## Conclusions

This project demonstrated the success of Random Forest in classifying mushrooms with perfect accuracy. Key factors that contributed to this result included:
- Thorough data cleaning and preprocessing.
- Detailed visualization to identify key predictive features.
- Careful evaluation and selection of machine learning models.

## Future Work

Potential areas for further exploration include:
- Experimenting with hyperparameter tuning for models like Random Forest and Support Vector Machines.
- Investigating feature importance to better interpret how features like `odor` and `spore-print-color` influence predictions.
- Applying this approach to similar classification tasks for broader validation.

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

