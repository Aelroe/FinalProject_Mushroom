# Mushroom Classification Project

![](UTA-DataScience-Logo.png)

## Mushroom Classification Project

This repository holds an attempt to classify mushrooms as edible or poisonous using the Mushroom Classification Dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification).

## Overview

The task is to classify mushrooms as either edible (e) or poisonous (p) based on their features, such as cap shape, color, and odor, provided in the dataset.

I treated this problem as a classification task. Several machine learning models were trained and compared, including Logistic Regression, Random Forest, K-Nearest Neighbors, and Support Vector Machines. Preprocessing involved cleaning and one-hot encoding categorical features.

My best model, Random Forest, achieved perfect accuracy (100%) on the validation set.

## Summary of Workdone

### Data

* **Data:**
  * **Input:** A CSV file containing features of mushrooms (22 categorical features) and the target column (`class`).
  * **Size:** 8,124 rows and 23 columns.
  * **Instances:**
    * Train: 70%
    * Validation: 15%
    * Test: 15%

#### Preprocessing / Clean up

* Dropped irrelevant features, such as `veil-type`, which had only one value.
* Replaced `?` in `stalk-root` with `unknown` to handle missing data.
* One-hot encoded categorical features, excluding the target column.
* Ensured that no duplicate columns were present.

#### Data Visualization

* Histograms were created to compare feature distributions across edible and poisonous mushrooms.
* Tabular summaries for categorical features were provided for better interpretability.

### Problem Formulation

* **Input / Output:**
  * Input: Preprocessed features after one-hot encoding.
  * Output: Binary classification of mushrooms (`e` = edible, `p` = poisonous).
* **Models:**
  * Logistic Regression
  * Random Forest
  * K-Nearest Neighbors
  * Support Vector Machines
* **Metrics:** Accuracy and classification reports were used to evaluate the models.

### Training

* **Description:**
  * Software: Python, Jupyter Notebook.
  * Libraries: pandas, scikit-learn, matplotlib.
  * Hardware: MacBook.
  * Models were trained on the training set and evaluated on the validation set.
* **Stopping Criteria:**
  * Training continued until all models were compared, and the best-performing model was selected based on accuracy.

### Performance Comparison

* **Metrics:** Accuracy, precision, recall, and F1-score.
* **Results:**
  * Random Forest achieved 100% accuracy.
  * Logistic Regression achieved 99.75% accuracy.
  * K-Nearest Neighbors and Support Vector Machines also achieved 100% accuracy.
* **Conclusion:** Random Forest was selected as the final model due to its simplicity and performance.

### Conclusions

* The Random Forest model is highly effective for this classification task, achieving perfect accuracy.
* Preprocessing and one-hot encoding were critical to the success of the models.

### Future Work

* Experiment with additional models and hyperparameter tuning to further optimize performance.
* Use advanced visualization techniques to analyze feature importance in the Random Forest model.
* Apply the methodology to similar datasets to test generalizability.

## How to Reproduce Results

### Steps:

1. **Preprocessing:**
   * Run the provided Jupyter Notebook (`Final_Project.ipynb`) to clean and preprocess the data.
2. **Training:**
   * Execute the training steps for all models.
3. **Evaluation:**
   * Use the evaluation code to generate metrics and validate model performance.
4. **Submission:**
   * Apply the trained model to the test set and generate a submission file (`submission.csv`).

### Overview of Files in Repository

* **Files:**
  * `Final_Project.ipynb`: Jupyter Notebook containing all steps of the project.
  * `submission.csv`: Final predictions for the Kaggle challenge.
  * `README.md`: Project description and documentation.

## Software Setup

* **Required Packages:**
  * pandas
  * scikit-learn
  * matplotlib
* **Installation:**
  * Install the packages using pip:
    ```bash
    pip install pandas scikit-learn matplotlib
    ```

## Citations

* Mushroom Classification Dataset: [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification).
* Scikit-learn documentation: https://scikit-learn.org/stable/index.html.

