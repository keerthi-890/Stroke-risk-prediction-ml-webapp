A Web-Based Interface That Leverages Machine Learning to Assess an Individual’s Vulnerability to Brain Stroke

Abstract

Stroke is a major global health concern and a leading cause of mortality and long-term disability. 
Early identification of individuals at high risk can significantly improve preventive outcomes. 
This project presents a machine learning-based stroke risk prediction system integrated into 
a web-based interface. The system employs Logistic Regression with SMOTE oversampling and Standard 
Scaling to address class imbalance and improve minority-class detection. The trained model is 
deployed using Streamlit to provide a user-friendly interface for real-time stroke risk assessment. 
The application emphasizes interpretability, accessibility, and computational efficiency.


Keywords

Machine Learning, 
Stroke Prediction, 
Logistic Regression, 
SMOTE, 
Standard Scaler, I
mbalanced Dataset, 
Explainable AI, 
SHAP, Streamlit, 
Health Informatics

I. Introduction

Stroke (Cerebrovascular Accident) occurs when blood supply to the brain is interrupted due to blockage or hemorrhage, 
leading to brain tissue damage. Predicting stroke risk using machine learning techniques can assist in early 
intervention and preventive healthcare.

This project develops a predictive model trained on healthcare data and deploys it through a web interface, 
enabling non-technical users to assess stroke vulnerability based on personal health parameters.


II. Dataset Description

The system utilizes the Healthcare Stroke Dataset containing 5,110 patient records with 11 attributes including:

- Gender  
- Age  
- Hypertension  
- Heart Disease  
- Ever Married  
- Work Type  
- Residence Type  
- Average Glucose Level  
- BMI  
- Smoking Status  
- Stroke (Target Variable)  

The dataset exhibits severe class imbalance, with approximately 5% stroke-positive cases.


III. Methodology

A. Data Preprocessing

1. Removed irrelevant column (id)  
2. Handled missing BMI values using median imputation  
3. Applied Label Encoding for binary categorical variables  
4. Applied One-Hot Encoding for nominal variables  
5. Performed feature scaling using StandardScaler  

B. Handling Class Imbalance

Synthetic Minority Oversampling Technique (SMOTE) was applied to the training dataset to balance stroke-positive cases.

C. Model Training

- Logistic Regression  
- Hyperparameter tuning using RandomizedSearchCV  
- Threshold optimization based on F1-score  

D. Model Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Confusion Matrix  

IV. Explainable Artificial Intelligence

To enhance interpretability, SHAP (SHapley Additive Explanations) was implemented to:

- Determine feature importance  
- Visualize feature impact on predictions  
- Improve model transparency  


V. System Architecture

1. Data Preprocessing  
2. Feature Scaling  
3. SMOTE Oversampling  
4. Logistic Regression Training  
5. Model Saving (Pickle)  
6. Deployment using Streamlit Web Interface  


VI. Web Application

The application is developed using Streamlit and allows users to:

- Input health parameters  
- Obtain real-time stroke risk prediction  
- View prediction results instantly  

No user data is stored; predictions are processed in real-time to ensure privacy.

VII. Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Matplotlib & Seaborn  
- SHAP  
- Streamlit  
- Pickle  

VIII. Installation and Execution

Step 1: Clone Repository
        git clone https://github.com/your-username/stroke-prediction-app.git  
        cd stroke-prediction-app  

Step 2: Install Dependencies
        pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn shap streamlit  

Step 3: Train Model
        python train1.py  

Step 4: Run Web Application
        streamlit run stroke_app.py  

IX. Results

The optimized Logistic Regression model with SMOTE and Standard Scaling achieved high predictive performance, 
particularly improving recall and F1-score for the minority (stroke) class, making it suitable for healthcare risk assessment.

X. Conclusion

This project successfully demonstrates the development of an interpretable and efficient 
machine learning-based stroke risk prediction system. By integrating predictive modeling with a
user-friendly web interface, the system bridges the gap between advanced machine learning techniques a
nd practical healthcare applications. The solution serves as a supportive tool for early risk screening 
but is not intended to replace professional medical diagnosis.


Author

Keerthika M  
Master of Computer Applications (MCA)  
Final Year Mini Project  
Machine Learning & Health Informatics  


## Disclaimer

This application is developed for educational and research purposes only. It should not be used as a substitute for professional medical advice or diagnosis.
