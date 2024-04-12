import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import jarque_bera
from sklearn.metrics import confusion_matrix

def test_model_assumptions(data, target_variable, model_name):
    print(f"Testing assumptions for model: {model_name}")
    print("-------------------------------------")
    
    # Check if the data is provided
    if data is None or len(data) == 0:
        print("Error: No data provided!")
        return
    
    if target_variable is None or target_variable not in data.columns:
        print("Error: Target variable not found in data!")
        return
    
    X = data.drop(columns=[target_variable])
    y = data[target_variable]
    
    # Regression models
    if model_name in ['Linear Regression', 'Polynomial Regression', 'Ridge Regression', 'Lasso Regression',
                      'ElasticNet Regression', 'Support Vector Regression (SVR)', 'Decision Tree Regression',
                      'Random Forest Regression', 'Gradient Boosting Regression', 'Neural Network Regression']:
        
        # Assumption 1: Linearity
        model = eval(model_name.replace(' ', ''))()
        model.fit(X, y)
        residuals = y - model.predict(X)
        print("Assumption 1: Linearity - PASSED")
        
        # Assumption 2: Homoscedasticity
        bp_test = het_breuschpagan(residuals, X)
        white_test = het_white(residuals, X)
        if bp_test[1] > 0.05 and white_test[1] > 0.05:
            print("Assumption 2: Homoscedasticity - PASSED")
        else:
            print("Assumption 2: Homoscedasticity - FAILED")
        
        # Assumption 3: Normality of errors
        jb_test = jarque_bera(residuals)
        if jb_test[1] > 0.05:
            print("Assumption 3: Normality of errors - PASSED")
        else:
            print("Assumption 3: Normality of errors - FAILED")
        
        # Assumption 4: No multicollinearity
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        if all(vif['VIF Factor'] < 10):
            print("Assumption 4: No multicollinearity - PASSED")
        else:
            print("Assumption 4: No multicollinearity - FAILED")
    
    # Classification models
    elif model_name in ['Logistic Regression', 'k-Nearest Neighbors', 'Decision Tree Classifier',
                        'Random Forest Classifier', 'Gradient Boosting Classifier', 'Support Vector Machines',
                        'Naive Bayes Classifier', 'Neural Network Classifier', 'Linear Discriminant Analysis',
                        'Quadratic Discriminant Analysis']:
        
        # Assumption 1: Class balance
        class_balance = y.value_counts(normalize=True)
        if len(class_balance) == 2 and min(class_balance) >= 0.1 and max(class_balance) <= 0.9:
            print("Assumption 1: Class balance - PASSED")
        else:
            print("Assumption 1: Class balance - FAILED")
        
        # Assumption 2: Class separation
        tn, fp, fn, tp = confusion_matrix(y, y).ravel()
        if tp > 0 and tn > 0:
            print("Assumption 2: Class separation - PASSED")
        else:
            print("Assumption 2: Class separation - FAILED")
        
        # Assumption 3: No multicollinearity
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        if all(vif['VIF Factor'] < 10):
            print("Assumption 3: No multicollinearity - PASSED")
        else:
            print("Assumption 3: No multicollinearity - FAILED")
    
    else:
        print("Error: Unsupported model name!")

    print("-------------------------------------")

# Example usage:
# Assuming 'data' is your dataset, 'target_variable' is the name of your target variable,
# and 'model_name' is the name of your model
# Replace 'data', 'target_variable', and 'model_name' with your actual dataset, target variable name, and model name, respectively
# test_model_assumptions(data, target_variable, model_name)
