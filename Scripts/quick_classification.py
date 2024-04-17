import numpy as np 
import pandas as pd 
import time
import warnings
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
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
    Provide lists of classification classifiers and their names.
    """
    n_jobs = -1
    random_state = 1

    classifiers = [
        DummyClassifier(), 
        LogisticRegression(n_jobs=n_jobs), 
        SGDClassifier(random_state=random_state),
        SVC(kernel="linear", probability=True, random_state=random_state),
        LinearSVC(random_state=random_state),
        DecisionTreeClassifier(random_state=random_state),
        RandomForestClassifier(n_jobs=n_jobs, random_state=random_state),
        GradientBoostingClassifier(random_state=random_state),
        lgb.LGBMClassifier(n_jobs=n_jobs, random_state=random_state, verbosity=-1),
        xgb.XGBClassifier(n_jobs=n_jobs, random_state=random_state, verbosity=0),
    ]

    clf_names = [
        "DummyClassifier          ",
        "LogisticRegression       ", 
        "SGDClassifier            ",
        "SVC                      ",
        "LinearSVC                ",
        "DecisionTreeClassifier   ",
        "RandomForestClassifier   ", 
        "GradientBoostingClassifier", 
        "LGBMClassifier           ", 
        "XGBClassifier            ",
    ]

    return clf_names, classifiers

def prepare_data(df, target_name):
    """
    Separate descriptive variables and target variable.
    Separate numerical and categorical columns.
    """
    if isinstance(df, pd.DataFrame):
        if target_name is not None:
            X = df.drop(target_name, axis=1)
            y = df[target_name]
        else:
            X = df
            y = None

        num_cols = X.select_dtypes("number").columns
        cat_cols = X.select_dtypes("object").columns
    elif isinstance(df, np.ndarray):
        if target_name is not None:
            X = np.delete(df, target_name, axis=1)
            y = df[:, target_name]
        else:
            X = df
            y = None

        num_cols = None  # Unable to extract column names from numpy arrays
        cat_cols = None  # Unable to extract column names from numpy arrays

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
        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', transformed_classifier)])
    else:
        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])


def score_models(df, target_name, sample_size=None, 
    impute_strategy="mean", scoring_metrics=("accuracy",), verbose=True):
    """
    Score classification models.
    """
    if sample_size is not None:
        df = df.sample(sample_size)
  
    X, y, num_cols, cat_cols = prepare_data(df, target_name)

    scores = []

    clf_names, classifiers = get_classifiers()
    if verbose:
        print(f"Classifier             Metrics")
        print("-" * 60)
    for clf_name, classifier in zip(clf_names, classifiers):
        start_time = time.time()
        clf = get_pipeline(classifier, num_cols, cat_cols, impute_strategy, log_x=False, log_y=False)
        clf.fit(X, y)

        y_pred = clf.predict(X)

        metric_scores = {}
        for metric in scoring_metrics:
            if metric == "accuracy":
                metric_scores[metric] = accuracy_score(y, y_pred)
            elif metric == "precision":
                metric_scores[metric] = precision_score(y, y_pred)
            elif metric == "recall":
                metric_scores[metric] = recall_score(y, y_pred)
            elif metric == "f1":
                metric_scores[metric] = f1_score(y, y_pred)
            elif metric == "roc_auc":
                if hasattr(classifier, "decision_function"):
                    y_score = classifier.decision_function(X)
                else:
                    y_score = classifier.predict_proba(X)[:, 1]
                metric_scores[metric] = roc_auc_score(y, y_score)

        if verbose:
            print(f"{clf_name.strip()} {', '.join([f'{k}: {v:.4f}' for k, v in metric_scores.items()])} | {(time.time() - start_time):.2f} secs")

        scores.append([clf_name.strip()] + [metric_scores[metric] for metric in scoring_metrics])

    scores_df = pd.DataFrame(scores, columns=["Classifier"] + list(scoring_metrics))
    return scores_df

# Example usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate some sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target_variable"] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop("target_variable", axis=1), df["target_variable"], test_size=0.2, random_state=42)
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

# Score models
scores = score_models(df_train, "target_variable", sample_size=None, 
                      impute_strategy="mean", scoring_metrics=("accuracy", "precision", "recall", "f1", "roc_auc"), verbose=True)
print("\nScores for Classification Models:\n", scores)
