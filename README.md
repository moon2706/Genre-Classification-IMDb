# Genre Classification using CNN on IMDb Dataset

## 📌 **Project Overview**
This project focuses on classifying movie genres using **Convolutional Neural Networks (CNN)**. The dataset is derived from IMDb, containing movie titles, descriptions, and genres. The primary objective is to build an effective genre classification model using deep learning.

---

## 🛠️ **Objectives**
- Perform data preprocessing and text cleaning.
- Implement a CNN model for text classification.
- Achieve high accuracy using efficient deep learning techniques.
- Evaluate model performance using accuracy metrics.

---

## 📁 **Dataset Overview**
The dataset can be downloaded from [Kaggle - Genre Classification Dataset IMDb](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb).

The dataset consists of four files:
- **`train_data.txt`**: Contains the training data with columns - `ID`, `Title`, `Genre`, and `Description`.
- **`test_data.txt`**: Contains test data with `ID`, `Title`, and `Description`.
- **`description.txt`**: Dataset description.
- **`test_data_solution.txt`**: Contains the actual genres for test data.

### **Data Format**
- **Train Data**: `ID ::: TITLE ::: GENRE ::: DESCRIPTION`
- **Test Data**: `ID ::: TITLE ::: DESCRIPTION`

---

## 🚀 **How to Run the Project**

### **Prerequisites**
Ensure you have the following installed:
- Python 3.8 or higher
- TensorFlow
- NumPy
- Pandas
- Scikit-Learn

### **Steps to Run**
1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/Genre-Classification-IMDb.git
    cd Genre-Classification-IMDb
    ```
2. Install dependencies (Optional - If you have a requirements file):
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, install individual packages:
    ```bash
    pip install tensorflow numpy pandas scikit-learn
    ```
3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
4. Execute the cells in the notebook sequentially.
5. Model training progress and evaluation results will be displayed.

---

## 🧑‍💻 **Code Overview**

1. **Data Loading & Preprocessing**
    - Reads data using Pandas.
    - Normalizes text (lowercasing).
    - Encodes labels using `LabelEncoder`.
2. **Tokenization and Padding**
    - Converts text into sequences using `Tokenizer`.
    - Pads sequences to ensure uniform input size.
3. **Model Building**
    - A CNN model with Conv1D and GlobalMaxPooling1D layers.
    - Fully connected Dense layers for classification.
4. **Training**
    - Trains the model using `SparseCategoricalCrossentropy` as the loss function.
5. **Evaluation**
    - Evaluates training accuracy using `model.evaluate()`.

---

## 📊 **Evaluation Criteria**
- **Functionality:** Model is implemented using CNN with effective text classification.
- **Accuracy:** Evaluated using training data.
- **Readability:** Code is clean, well-commented, and structured.
- **Documentation:** Comprehensive README and comments.

### **Model Performance Comparison**
Here are the accuracy scores of different models applied to the dataset:

#### **Machine Learning Models:**
- Logistic Regression: **58.62%**
- LightGBM: **56.77%**
- XGBoost: **54.38%**
- Naive Bayes: **51.01%**
- Random Forest: **47.59%**
- KNN: **41.23%**
- Decision Tree: **33.59%**

#### **Deep Learning Models:**
- CNN: **89.78%** *(Selected Model)*
- GRU: **74.61%**
- LSTM: **72.24%**

Based on the results, the CNN model was selected as it provided the highest accuracy.

---

## 🛡️ **Future Enhancements**
- Implement LSTM, GRU, or Transformers for better accuracy.
- Perform hyperparameter tuning.
- Integrate BERT for contextual understanding.
- Create a web-based UI for predictions.

---

## 🙌 **Contributors**
- **Name** - RISHIKESH PRADHAN
- Feel free to submit issues and contribute to the project.
