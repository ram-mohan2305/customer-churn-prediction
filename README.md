# customer-churn-prediction
A Logistic Regression &amp; Random Forest Model to predict the possibility of a customer exiting using the customer churn dataset available at https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data

**# Customer Churn Prediction Project Guide**

## **Task Objectives**
- Analyze historical customer data to predict churn.
- Consider key factors like usage patterns, demographics, and subscription duration.
- Handle missing values using appropriate imputation strategies.
- Build a machine learning model to effectively identify customers likely to churn.
- Extract insights into the most significant factors contributing to churn.

---

## **Steps to Run the Project**

### **1. Set Up Environment**
- Install required dependencies:
  ```sh
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```
- Open the Jupyter Notebook:
  ```sh
  jupyter notebook Customer_Churn_Prediction.ipynb
  ```

### **2. Load the Dataset**
- Read the customer churn dataset into a Pandas DataFrame.
- Perform exploratory data analysis (EDA) to understand trends and missing values.

### **3. Data Preprocessing**
- Handle missing values using appropriate imputation strategies.
- Encode categorical variables into numerical format.
- Normalize or standardize numerical features if necessary.

### **4. Exploratory Data Analysis (EDA)**
- Generate summary statistics and visualizations.
- Identify correlations between features and churn.
- Create graphs such as:
  - Correlation matrix
  - Line charts for feature trends
  - Pie charts for gender distribution in churned customers

### **5. Model Training & Evaluation**
- Split the dataset into training and testing sets.
- Train a machine learning model (e.g., Logistic Regression, Random Forest).
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Optimize hyperparameters for better prediction accuracy.

### **6. Generate Insights & Interpretation**
- Identify key factors driving customer churn.
- Analyze how balance, tenure, and demographics affect churn.
- Provide actionable recommendations based on findings.

### **7. Save & Deploy Model (Optional)**
- Save the trained model using joblib or pickle.
- Deploy it for real-time predictions if needed.

---

## **Code Quality Guidelines**
- **Use Clear and Descriptive Variable Names:** Avoid generic names like `df1`, `x`, `y`. Instead, use `customer_data`, `churn_rate`, etc.
- **Comment Code Properly:**
  ```python
  # Load dataset and display first five rows
  df = pd.read_csv("Churn_Modelling.csv")
  print(df.head())
  ```
- **Organize Code into Sections:** Use headings and markdown cells in Jupyter Notebook.
- **Optimize Performance:** Avoid redundant calculations, and use vectorized operations with Pandas.
