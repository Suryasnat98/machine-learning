🚀 Rock vs Mine Detection using Logistic Regression

A Machine Learning project that classifies sonar signals as either Rock (R) or Mine (M) using Logistic Regression.

This project demonstrates the complete ML workflow — from data preprocessing to building a predictive system.

📌 Project Overview

Sonar signals reflect differently from rocks and metal cylinders (mines).
By analyzing 60 numerical frequency-based features, we can train a model to classify the object.

This project uses:

NumPy

Pandas

Scikit-learn

📊 Dataset Information

Dataset: Sonar Dataset

Total Features: 60 numerical attributes

Target Classes:

R → Rock

M → Mine

Train-Test Split: 80% Training, 20% Testing

Stratified Sampling used to maintain class balance

🧠 Machine Learning Workflow
1️⃣ Data Loading

Dataset loaded using Pandas.

2️⃣ Feature & Target Separation

X → Features (60 columns)

Y → Target column

3️⃣ Train-Test Split
train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


Stratification ensures both Rock and Mine classes are evenly distributed.

4️⃣ Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)

5️⃣ Model Evaluation

Training Accuracy: (Add your printed value here)

Test Accuracy: (Add your printed value here)

Accuracy calculated using:

accuracy_score()

🔍 Predictive System

A custom prediction system is built where:

User inputs 60 feature values

Data is reshaped into a 2D array

Model predicts whether the object is Rock or Mine

Example:

prediction = model.predict(input_data_reshaped)


Output:

The object is a Rock


or

The object is a Mine

🖥️ How to Run This Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/rock-vs-mine.git
cd rock-vs-mine

2️⃣ Install Dependencies
pip install numpy pandas scikit-learn

3️⃣ Run the Script
python sonar_model.py

🎯 Key Learnings

Understanding Logistic Regression for binary classification

Importance of stratified train-test splitting

Model evaluation using accuracy score

Converting 1D input into 2D format for predictions

Building a simple ML prediction system

🚀 Future Improvements

Apply Feature Scaling (StandardScaler)

Use Cross-Validation

Try advanced models (SVM, Random Forest)

Add Confusion Matrix & Classification Report

Deploy as a Web App (Streamlit / Flask)
