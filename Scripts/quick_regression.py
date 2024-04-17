import numpy as np 
import pandas as pd 
import time
import warnings
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score, KFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, BayesianRidge, SGDRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

def log_transform(x):
    """Log transform function."""
    return np.log1p(x)

def inverse_log_transform(x):
    """Inverse log transform function."""
    return np.expm1(x)

def get_classifiers():
    """
    Provide lists of regression classifiers and their names.
    """
    n_jobs = -1
    random_state = 1

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
        lgb.LGBMRegressor(n_jobs=n_jobs, random_state=random_state, verbosity=-1),
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

    num_cols = X.select_dtypes("number").columns
    cat_cols = X.select_dtypes("object").columns
    
    return X, y, num_cols, cat_cols

def get_pipeline(classifier, num_cols, cat_cols, impute_strategy, log_x, log_y):
    """
    Create a pipeline for preprocessing and modeling.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=impute_strategy)),
        ('scaler', StandardScaler()),
    ])

    if log_x:
        numeric_transformer.steps.insert(1, ('log_transform', FunctionTransformer(log_transform)))

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
        ])

    if log_y:
        transformed_classifier = TransformedTargetRegressor(regressor=classifier, 
            func=log_transform, inverse_func=inverse_log_transform)
        return Pipeline(steps=[('preprocessor', preprocessor), 
                               ('selector', VarianceThreshold()), 
                               ('classifier', transformed_classifier)])
    else:
        return Pipeline(steps=[('preprocessor', preprocessor), 
                               ('selector', VarianceThreshold()), 
                               ('classifier', classifier)])

def score_models(df, target_name, sample_size=None, 
    impute_strategy="mean", scoring_metrics=("r2", "mae", "mse", "rmse"), log_x=False, log_y=False, verbose=True):
    """
    Score regression models.
    """
    if sample_size is not None:
        df = df.sample(sample_size)
  
    X, y, num_cols, cat_cols = prepare_data(df, target_name)

    scores = {metric: [] for metric in scoring_metrics}

    clf_names, classifiers = get_classifiers()
    if verbose:
        print("Classifier             Metrics")
        print("-" * 50)
    for clf_name, classifier in zip(clf_names, classifiers):
        start_time = time.time()
        clf = get_pipeline(classifier, num_cols, cat_cols, impute_strategy, log_x, log_y)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            for metric in scoring_metrics:
                if metric == "r2":
                    score = r2_score(y_test, y_pred)
                elif metric == "mae":
                    score = mean_absolute_error(y_test, y_pred)
                elif metric == "mse":
                    score = mean_squared_error(y_test, y_pred)
                elif metric == "rmse":
                    score = np.sqrt(mean_squared_error(y_test, y_pred))
                scores[metric].append(score)
        mean_scores = {metric: np.mean(scores[metric]) for metric in scoring_metrics}
        if verbose:
            print(f"{clf_name} {', '.join([f'{metric}: {mean_scores[metric]: .4f}' for metric in scoring_metrics])}  |  {(time.time() - start_time):.2f} secs")
        
    scores_df = pd.DataFrame(scores)
    scores_df.loc["mean_all"] = scores_df.mean()
    return scores_df

def train_models(df, target_name, 
    impute_strategy="mean", log_x=False, log_y=False, verbose=True): 
    """
    Train regression models.
    """
    X, y, num_cols, cat_cols = prepare_data(df, target_name)

    pipelines = []

    if verbose:
        print("Classifier            Training time")
        print("-" * 35)
    
    clf_names, classifiers = get_classifiers()
    for clf_name, classifier in zip(clf_names, classifiers):
        start_time = time.time()
        clf = get_pipeline(classifier, num_cols, cat_cols, impute_strategy, log_x, log_y)
        clf.fit(X, y)
        if verbose:
            print(f"{clf_name}     {(time.time() - start_time):.2f} secs")
        pipelines.append(clf)
    
    return pipelines

def predict_from_models(df_test, pipelines):
    """
    Make predictions using trained models.
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

def get_feature_importance(pipelines, feature_names):
    """
    Get feature importances for models supporting it.
    """
    feature_importances = {}
    for pipeline in pipelines:
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            importances = pipeline.named_steps['classifier'].feature_importances_
            if len(importances) == len(feature_names):
                feature_importances[type(pipeline.named_steps['classifier']).__name__] = dict(zip(feature_names, importances))
    return feature_importances


# Example usage
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate some sample data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target_variable"] = y

# Automated feature selection
selector = VarianceThreshold()
X_selected = selector.fit_transform(df.drop("target_variable", axis=1))
selected_columns = df.drop("target_variable", axis=1).columns[selector.get_support()]
df = pd.DataFrame(X_selected, columns=selected_columns)
df["target_variable"] = y

# Score models
scores = score_models(df, "target_variable", sample_size=None, 
                      impute_strategy="mean", scoring_metrics=("r2", "mae", "mse", "rmse"), log_x=False, log_y=False, verbose=True)
print("\nScores for Regression Models:\n", scores)

# Train models
pipelines = train_models(df, "target_variable", 
                         impute_strategy="mean", log_x=False, log_y=False, verbose=True)

# Get feature importances
feature_importances = get_feature_importance(pipelines, df.drop("target_variable", axis=1).columns)
print("\nFeature Importances:\n", feature_importances)

# Make predictions
predictions = predict_from_models(df_test.drop("target_variable", axis=1), pipelines)
print("\nPredictions from Regression Models:\n", predictions.head())
