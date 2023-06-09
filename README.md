# Loan-Prediction

**Loan Prediction Model**
This project aims to predict the loan status of individuals based on various features using a logistic regression model. The code provided demonstrates the steps involved in building and evaluating the model.

**Dataset
The dataset used for this project consists of two files:**
        train.csv: Contains the training data used to build the model.
        test.csv: Contains the test data used to make predictions.
        
**Kaggle Link of the dataset**
        https://www.kaggle.com/code/sazid28/home-loan-prediction

**The code requires the following dependencies:**
      numpy: A library for numerical computations in Python.
      pandas: A library for data manipulation and analysis.
      missingno: A library for visualizing missing data.
      matplotlib: A library for creating visualizations in Python.
      seaborn: A library for statistical data visualization.
      scikit-learn: A library for machine learning algorithms in Python.
**Usage**
      Clone the repository and navigate to the project directory.
      Place the train.csv and test.csv files in the project directory.
      Run the code using a Python IDE or Jupyter Notebook. 
      
**Code Explanation**
  *The code begins by importing the necessary libraries and loading the dataset into pandas DataFrames (loan_train and loan_test).
  *Basic exploratory data analysis is performed, including displaying the first few rows of the training data, checking the shape of the dataset, and obtaining descriptive statistics.
  *The explore_object_type function is defined to explore the values and counts of categorical features.
  *Categorical features are explored using the explore_object_type function.
  *Missing data is visualized using the missingno library.
  *Missing values are handled by imputing the mode for categorical features (Gender, Dependents, Married) and the mean for numerical features (LoanAmount, Credit_History).
  *Label encoding is applied to convert categorical features into numerical representations.
  *Data visualization is performed using matplotlib and seaborn to explore the distribution of loan application amounts.
  *The logistic regression model is created using scikit-learn's LogisticRegression class.
  *The model is trained on the training data using the specified features (Property_Area, Education, ApplicantIncome, LoanAmount, Credit_History) and the loan status (Loan_Status) as the target variable.
  *The model is evaluated by calculating the accuracy score on the training data.
  *Finally, the model is used to make predictions on the test data.

**Results**
The logistic regression model achieved an accuracy score of approximately 80.78% on the training data.
This can be tried using Random Forest and SVM too.

