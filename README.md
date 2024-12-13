# Mushroom Classification Project

![](UTA-DataScience-Logo.png)

## Mushroom Classification Project

This project focuses on classifying mushrooms as either edible or poisonous using the Mushroom Classification Dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification). Using machine learning, the goal was to develop an accurate model to classify mushrooms based on their features.

## Overview

The dataset contains 8,124 rows and 22 categorical features describing mushrooms, such as cap shape, color, and odor. The target column (`class`) specifies whether a mushroom is edible (`e`) or poisonous (`p`).

To achieve 100% classification accuracy, Random Forest was selected as the final model after comparing it with Logistic Regression, K-Nearest Neighbors, and Support Vector Machines. This result was achieved by combining detailed data exploration, preprocessing, feature visualization, and model evaluation.

## Summary of Work

### Data Exploration

Key insights from exploration:
- The `veil-type` column had only one unique value and was removed to simplify the dataset.
- The `stalk-root` column contained missing values (`?`), which were replaced with "unknown" to ensure no data loss.
- **Odor** was identified as the most significant feature for distinguishing between edible and poisonous mushrooms, showing clear differences between the two classes.
- Class distributions were nearly balanced, making this a well-suited dataset for classification tasks.

![image](https://github.com/user-attachments/assets/e3491a74-2994-4970-b0c7-1a82df902d59)

### Data Cleaning and Preprocessing

Data preparation was an essential step to make the dataset ready for machine learning:
- One-hot encoding was applied to all categorical features, ensuring compatibility with machine learning models.
- The target column (`class`) was encoded as `0` for edible and `1` for poisonous mushrooms.
- To address missing data in `stalk-root`, the value `'?'` was replaced with "unknown."
- Irrelevant features like `veil-type`, which added no variability, were dropped.

These preprocessing steps ensured the data was clean, consistent, and suitable for robust model training.

### Data Visualization

Visualizations were critical in understanding the dataset and shaping the model:
- **Feature Importance:** A Random Forest model highlighted the top 15 most important features, with `odor` and `gill-size` being the most predictive.

![image](https://github.com/user-attachments/assets/f8825499-6ed6-4e06-8bc7-d8a1ff673625)

- **Odor Feature:** The `odor` feature strongly distinguished between edible and poisonous mushrooms, showcasing a clear predictive pattern.

![image](https://github.com/user-attachments/assets/bd7b8984-0ff0-4c27-bb04-14d6111baf24)

### Problem Formulation

This project was framed as a binary classification problem:
- **Input:** Preprocessed categorical features after one-hot encoding.
- **Output:** Binary classification of mushrooms as `0` (edible) or `1` (poisonous).

To ensure fairness in evaluation, the data was split into training, validation, and test sets in a 70-15-15 ratio using stratified sampling. This strategy preserved class balance across all subsets.

### Model Training and Comparison

We tested four machine learning models: Logistic Regression, Random Forest, K-Nearest Neighbors, and Support Vector Machines. Each model was trained on the training set and evaluated on the validation set. Random Forest emerged as the best model due to:
- Perfect accuracy on validation and test sets.
- Its ability to handle categorical features and provide feature importance insights.
- Robustness to noise and overfitting.

| Model                  | Validation Accuracy | Test Accuracy | Notes                                   |
|------------------------|---------------------|---------------|-----------------------------------------|
| Logistic Regression    | 99.75%             | 99.75%        | Simple and interpretable.               |
| Random Forest          | 100%               | 100%          | Robust and provided feature importance. |
| K-Nearest Neighbors    | 100%               | 100%          | Computationally expensive.              |
| Support Vector Machines| 100%               | 100%          | High accuracy but less interpretable.   |

### Test Set Evaluation

The final Random Forest model was evaluated on the test set and achieved 100% accuracy. This result confirmed the model's reliability and generalizability. Below is a preview of the predictions:

| Index | Prediction |
|-------|------------|
| 0     | 1          |
| 1     | 0          |
| 2     | 1          |
| ...   | ...        |

### Submission

The final predictions were saved in `submission.csv` for evaluation and sharing. The project's success lay in combining well-executed preprocessing, feature selection, and robust modeling.

## Conclusions

This project successfully classified mushrooms with perfect accuracy using Random Forest. Key factors that contributed to this success included:
- A clean, well-preprocessed dataset.
- Visualizations that highlighted predictive features like `odor` and `gill-size`.
- Careful model selection based on performance and interpretability.

## Future Work

Although the project achieved perfect accuracy, there are areas for further exploration:
- Hyperparameter tuning to optimize Random Forest and Support Vector Machines.
- Conducting feature selection experiments to analyze the importance of less influential features.
- Applying the methodology to similar datasets to test its generalizability.

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
