# https://www.kaggle.com/chmaxx/quick-regression/code
# https://www.kaggle.com/chmaxx/train-12-regressors-with-just-one-line-of-code

import numpy as np 
import pandas as pd 

from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import cross_val_score

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb
import lightgbm as lgb

import time

import warnings
warnings.filterwarnings('ignore')



def log_transform(x):
    return np.log1p(x)

def inverse_log_transform(x):
    return np.expm1(x)


def get_classifiers():

    """
    Provide lists of regression classifiers and their names.
    """
    n_jobs       = -1
    random_state =  1

    classifiers = [
                   DummyRegressor(), 
                   LinearRegression(n_jobs=n_jobs), 
                   Ridge(random_state=random_state), 
                   Lasso(random_state=random_state), 
                   ElasticNet(random_state=random_state),
                   KernelRidge(),
                   SGDRegressor(random_state=random_state),
                   SVR(kernel="linear"),
                   LinearSVR(random_state=1),
                   DecisionTreeRegressor(random_state=random_state),
                   RandomForestRegressor(n_jobs=n_jobs, random_state=random_state),
                   GradientBoostingRegressor(random_state=random_state),
                   lgb.LGBMRegressor(n_jobs=n_jobs, random_state=random_state),
                   xgb.XGBRegressor(objective="reg:squarederror", n_jobs=n_jobs, random_state=random_state),
    ]

    clf_names = [
                "DummyRegressor       ",
                "LinearRegression     ", 
                "Ridge                ",
                "Lasso                ",
                "ElasticNet           ",
                "KernelRidge          ",
                "SGDRegressor         ",
                "SVR                  ",
                "LinearSVR            ",
                "DecisionTreeRegressor",
                "RandomForest         ", 
                "GBMRegressor         ", 
                "LGBMRegressor        ", 
                "XGBoostRegressor     ",
    ]

    return clf_names, classifiers



def prepare_data(df, target_name):

    """
    Separate descriptive variables and target variable.
    Separate numerical and categorical columns.
    """

    if target_name is not None:
        X = df.drop(target_name, axis=1)
        y = df[target_name]
    else:
        X = df
        y = None

    # get list of numerical & categorical columns in order to process these separately in the pipeline 
    num_cols = X.select_dtypes("number").columns
    cat_cols = X.select_dtypes("object").columns
    
    return X, y, num_cols, cat_cols


def get_pipeline(classifier, num_cols, cat_cols, impute_strategy, log_x, log_y):

    """
    Create Pipeline with a separate pipe for categorical and numerical data.
    Automatically impute missing values, scale and then one hot encode.
    """

    # the numeric transformer gets the numerical data acording to num_cols
    # first step: the imputer imputes all missing values to the provided strategy argument
    # second step: all numerical data gets stanadard scaled 
    if log_x == False:
        numeric_transformer = Pipeline(steps=[
            ('imputer', make_pipeline(SimpleImputer(strategy=impute_strategy))),
            ('scaler', StandardScaler())])
    # if log_x is "True" than log transform feature values
    else:
        numeric_transformer = Pipeline(steps=[
            ('imputer', make_pipeline(SimpleImputer(strategy=impute_strategy))),
            ('log_transform', FunctionTransformer(np.log1p)),
            ('scaler', StandardScaler()),
            ])
    
    # the categorical transformer gets all categorical data according to cat_cols
    # first step: imputing missing values
    # second step: one hot encoding all categoricals
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # the column transformer creates one Pipeline for categorical and numerical data each
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)])
    
    # return the whole pipeline for the classifier provided in the function call
    if log_y == False:
        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    # if log_y is "True" than use a TransformedTargetRegressor with log and inverse log functions for "y"
    else:
        transformed_classifier = TransformedTargetRegressor(regressor=classifier, 
            func=log_transform, inverse_func=inverse_log_transform)
        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', transformed_classifier)])


def score_models(df, target_name, sample_size=None, 
    impute_strategy="mean", scoring_metric="r2", log_x=False, log_y=False, verbose=True):

    """
    This function yields error scores for a large variety of common regression classifiers on provided training data. 

    Function separates numerical and categorical data based on dtypes of the dataframe columns. Missing values are imputed. Categorical data is one hot encoded, numerical data standard scaled. All classifiers are used with default settings and crossvalidated.
    
    The function returns a dataframe with error scores of all classifiers as well as the mean of all results in the last row of the dataframe.

    Parameters
    ----------
    df : Pandas dataframe 
        Pandas dataframe with your training data
    target_name : str
        Column name of target variable
    sample_size : int, default "None" (score on all available samples)
        Number of samples for scoring the model
    impute_strategy : str, default "mean" 
        Strategy for SimpleImputer, can be "mean" (default), "median", "most_frequent" or "constant"
    scoring_metric : str, default "r2"
        scoring metric for regressor: "r2" (default), "explained_variance", "max_error", 
        "neg_mean_absolute_error", "neg_mean_squared_error", "neg_mean_squared_log_error", "neg_median_absolute_error"
    log_x : bool, default "False" 
        Log transform features variable(s)
    log_y : bool, default "False" 
        Log transform target variable
    verbose : bool, default "True" 
        Print results during crossvalidation
    
    Returns
    -------
    DataFrame
        1st column : Name of classifier
        2nd column : scoring result

    Example
    -------
        X, y = sklearn.datasets.make_regression()
        X, X_test, y, _ = train_test_split(X, y)

        df = pd.DataFrame(X)
        df["target_variable"] = y
        df_test = pd.DataFrame(X_test)

        scores_dummy = score_models(df, "target_variable")
        display(scores_dummy)
        
        # further use: train and predict
        pipelines = train_models(df, "target_variable")
        predictions = predict_from_models(df_test, pipelines)
        predictions.head()

    """

    
    if sample_size is not None:
        df = df.sample(sample_size)
  
    # retrieve X, y and separated columns names for numerical and categorical data
    X, y, num_cols, cat_cols = prepare_data(df, target_name)

    scores = []

    clf_names, classifiers = get_classifiers()
    if verbose == True:
        print(f"Classifier             Metric ({scoring_metric})")
        print("-"*30)
    for clf_name, classifier in zip(clf_names, classifiers):
        start_time = time.time()
        
        # create a pipeline for each classifier
        clf = get_pipeline(classifier, num_cols, cat_cols, impute_strategy, log_x, log_y)
                
        # crossvalidate classifiers on training data
        cv_score = cross_val_score(clf, X, y, cv=3, scoring=scoring_metric)
        
        if verbose == True:
            print(f"{clf_name} {cv_score.mean(): .4f}  |  {(time.time() - start_time):.2f} secs")
        
        scores.append([clf_name.strip(), cv_score.mean()])

    scores = pd.DataFrame(scores, columns=["Classifier", scoring_metric]).sort_values(scoring_metric, ascending=False)
    
    # just for good measure: add the mean of all scores to dataframe
    scores.loc[len(scores) + 1, :] = ["mean_all", scores[scoring_metric].mean()]

    return scores.reset_index(drop=True)
    


def train_models(df, target_name, 
    impute_strategy="mean", log_x=False, log_y=False, verbose=True): 

    """
    This function trains a large variety of common regression classifiers on provided training data. It separates numerical and categorical data based on dtypes of the dataframe columns. Missing values are imputed. Categorical data is one hot encoded, numerical data standard scaled. Each classifier is then trained with default settings.
    
    The function returns a list of fitted scikit-learn Pipelines.

    Parameters
    ----------
    df : Pandas dataframe 
        Pandas dataframe with your training data
    target_name : str
        Column name of target variable
    sample_size : int, default "None" (score on all available samples)
        Number of samples for scoring the model
    impute_strategy : str, default "mean" 
        Strategy for SimpleImputer, can be "mean" (default), "median", "most_frequent" or "constant"
    log_x : bool, default "False" 
        Log transform features variable(s)
    log_y : bool, default "False" 
        Log transform target variable
    verbose : bool, default "True" 
        Print results during crossvalidation
    
    Returns
    -------
    List of fitted scikit-learn Pipelines

    Example:
        X, y = sklearn.datasets.make_regression()
        X, X_test, y, _ = train_test_split(X, y)

        df = pd.DataFrame(X)
        df["target_variable"] = y
        df_test = pd.DataFrame(X_test)

        scores_dummy = score_models(df, "target_variable")
        display(scores_dummy)
        
        pipelines = train_models(df, "target_variable")

        # further use: predict from pipelines
        predictions = predict_from_models(df_test, pipelines)
        predictions.head()
    
    """

    X, y, num_cols, cat_cols = prepare_data(df, target_name)

    pipelines = []

    if verbose == True:
        print(f"Classifier            Training time")
        print("-"*35)
    
    clf_names, classifiers = get_classifiers()
    for clf_name, classifier in zip(clf_names, classifiers):
        start_time = time.time()
        clf = get_pipeline(classifier, num_cols, cat_cols, impute_strategy, log_x, log_y)
        clf.fit(X, y)
        if verbose == True:
            print(f"{clf_name}     {(time.time() - start_time):.2f} secs")
        pipelines.append(clf)
    
    return pipelines



def predict_from_models(df_test, pipelines):

    """
    This function makes predictions with a list of pipelines. Test data is treated in the same pipeline the classifiers were trained on. 
    
    The function returns a dataframe with all predictions ordered columnwise. Each column is named with the respective classifiers.

    Parameters
    ----------
    df_test : Pandas dataframe 
        Dataframe with test data
    pipelines: array
        List of scikit-learn pipelines (preferably from train_models())

    Returns
    -------
    Pandas dataframe with prediction from each classifier, ordered columnwise. 
    1 column = results of 1 classifier.
    
    Example:
        X, y = sklearn.datasets.make_regression()
        X, X_test, y, _ = train_test_split(X, y)

        df = pd.DataFrame(X)
        df["target_variable"] = y
        df_test = pd.DataFrame(X_test)

        scores_dummy = score_models(df, "target_variable")
        display(scores_dummy)
        
        pipelines = train_models(df, "target_variable")

        # further use: predict from pipelines
        predictions = predict_from_models(df_test, pipelines)
        predictions.head()
    
    """
    
    X_test, _ , _, _ = prepare_data(df_test, None)
    predictions = []
    
    for pipeline in pipelines:
        preds = pipeline.predict(X_test)
        predictions.append(preds)
        
    df_predictions = pd.DataFrame(predictions).T
    clf_names, _ = get_classifiers()
    df_predictions.columns = [clf_name.strip() for clf_name in clf_names]

    return df_predictions
