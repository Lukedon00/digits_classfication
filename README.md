# Digits Classification with K-Nearest Neighbors (KNN)

This project demonstrates the classification of the Digits dataset using the K-Nearest Neighbors (KNN) algorithm. The Digits dataset is loaded from scikit-learn's built-in datasets and used to train and evaluate a classification model.

## Project Overview

- **Dataset**: Digits dataset from `sklearn.datasets`
- **Algorithm**: K-Nearest Neighbors (k=3)
- **Objective**: Classify handwritten digits (0-9) based on pixel values.
- **Tools Used**: Python, Pandas, scikit-learn, Seaborn, Matplotlib

## Steps:
1. Load the Digits dataset.
2. Preprocess the data and create features and target variables.
3. Split the data into training and testing sets.
4. Train the model using K-Nearest Neighbors.
5. Evaluate the model using confusion matrix and classification report.

## Results:
- **Accuracy**: Achieved high accuracy in recognizing handwritten digits.
- **Confusion Matrix**:
    - [[x, y], [z, w], ...]
- **Classification Report**:  
    ```
    precision    recall  f1-score   support
    0       x.xx      x.xx      x.xx      xxx
    1       y.yy      y.yy      y.yy      yyy
    ```

## How to Run

### Prerequisites
- Python 3.x
- Required Libraries (see `requirements.txt`)

### Steps to Run:
1. Clone the repository.
    ```bash
    git clone https://github.com/your-username/digits_classification.git
    ```
2. Navigate to the project directory.
    ```bash
    cd digits_classification
    ```
3. Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebook or the Python script:
    - Jupyter Notebook: Open the `digits_classification.ipynb` file and run the cells.
    - Python Script: Execute the Python file in your terminal.
    ```bash
    python digits_classification.py
    ```

## License
This project is licensed under the MIT License.
