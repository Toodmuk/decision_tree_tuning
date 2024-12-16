# Decision Tree Classifier with Hyperparameter Tuning

This repository contains a complete workflow for building, tuning, and evaluating a decision tree classifier. The project focuses on predicting the success of space launches based on various features of the launches.

---

## Features of the Project
- **Data Preprocessing**: Cleans and prepares the dataset for machine learning.
- **Model Training**: Builds a decision tree classifier to predict launch success.
- **Hyperparameter Tuning**: Utilizes GridSearchCV to find the optimal model parameters.
- **Visualization**: Provides visualizations of the decision tree, feature importance, and confusion matrix.

---

## **Dataset Reference**
The dataset used for this project is:
- **Dataset Name**: Space Launches
- **Source**: Hugging Face
- **Reference**: https://huggingface.co/datasets/Tylerbry1/Historical_Space_Launch_Data_1957-2039/commit/6132d90437cf1e87fd41a2be4e57c9496ebb0591

---

## How to Use
1. **Requirements**: Install the required Python libraries:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
2.** Add Dataset**: Place the .xlsx file in the same directory as the Python script.

3. **Run the Script**. Execute the rocket.py scrit.

4. **Outputs**: Initial and tuned model accuracy / Decision tree visualization / Feature importance plot / Confusion matrix visualization
/ Decision tree rules saved in decision_tree_rules.txt
