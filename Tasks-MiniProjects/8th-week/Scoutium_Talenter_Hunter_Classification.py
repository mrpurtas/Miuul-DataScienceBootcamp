# Standard libraries for data manipulation and visualization
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Adjusting display settings for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Preprocessing and encoding tools
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Machine learning models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# Advanced ensemble models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Model evaluation and hyperparameter tuning tools
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier

# Tools for handling imbalanced data
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Import joblib for saving/loading models
import joblib

import warnings
warnings.filterwarnings("ignore")




# Reading a CSV file containing player attributes
# The data is separated by semicolons (';'), hence the use of sep=";"
attributes = pd.read_csv('datasets/machine_learning/scoutium_attributes.csv', sep=";")

# Reading another CSV file containing potential labels for the players
# Similar to the above, this file uses semicolons as separators and is located in the same directory
potential_labels = pd.read_csv('datasets/machine_learning/scoutium_potential_labels.csv', sep=";")


# Combining the attributes and potential labels into a single DataFrame.
# This merge operation aligns data based on shared columns: 'task_response_id', 'match_id', 'evaluator_id', and 'player_id'.
# The merge is a left join, ensuring all records from the attributes dataset are included, with missing entries in potential_labels filled as NaN.
df = pd.merge(attributes, potential_labels, on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'], how='left')


################################################
# 1. Exploratory Data Analysis
################################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


# Initial Data Exploration
# This function, check_df(df), is likely designed for initial exploration of the dataframe 'df'.
# It could include operations like checking for missing values, data types, and providing a quick overview of the dataset.
check_df(df)

# Categorization of Variable Types
# Using grab_col_names(df, cat_th=5, car_th=20) to categorize the columns of 'df' into different types:
# 'cat_cols' for categorical, 'num_cols' for numerical, and 'cat_but_car' for categorical but with many unique values.
# The thresholds for categorizing are set by cat_th and car_th parameters.
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

# Examination of Numerical Variables
# Generating descriptive statistics for numerical columns in 'df'.
# The .T at the end transposes the result for better readability and presentation.
df[num_cols].describe().T

# Correlation Analysis Among Numerical Variables
# Using correlation_matrix(df, num_cols) to calculate and visualize the correlation matrix for numerical columns in 'df'.
# This step is crucial for understanding the interrelationships among numerical variables.
correlation_matrix(df, num_cols)

# Analysis of Target Variable with Numerical Variables
# This loop iterates through each numerical column, applying the target_summary_with_num function.
# It analyzes the relationship between each numerical column and the target variable 'potential_label'.
# Such analysis is vital for understanding the influence of each numerical feature on the target variable.
for col in num_cols:
    target_summary_with_num(df, "potential_label", col)


################################################
# 2. Data Preprocessing & Feature Engineering
################################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# Outlier Detection in Numerical Columns
# This loop iterates through each numerical column in 'num_cols'.
# For each column, it applies the 'check_outlier' function to the dataframe 'df'.
# The function 'check_outlier' is likely designed to detect outliers in the specified column based on the given quantile thresholds.
# The thresholds are set at 0.05 and 0.95, meaning it checks for outliers outside the 5th and 95th percentiles.
# The loop prints the column name along with the result of the outlier check, indicating whether outliers are present.
for col in num_cols:
    print(col, check_outlier(df, col, 0.05, 0.95))


# Excluding Goalkeepers from the Classification
# In this step, we remove records of goalkeepers from the dataset.
# The 'position_id' column in 'df' represents player positions, where goalkeepers are denoted by the value 1.
# We use a boolean condition '(df["position_id"] != 1)' within df.loc to select all records where the position is not 1 (goalkeeper).
# This filters out the goalkeepers, ensuring they are not included in subsequent analysis or modeling processes.
df = df.loc[~(df["position_id"] == 1)]


# Handling Imbalanced Classes in 'potential_label'
# An examination of the 'potential_label' column's value distribution reveals a significant imbalance:
# - 'average': 8497 instances (79.19%)
# - 'highlighted': 2097 instances (19.54%)
# - 'below_average': 136 instances (1.27%)
# Noting the extremely low frequency of the 'below_average' class, we decide to remove these records from the dataset.
# This is achieved by filtering the DataFrame to exclude rows where 'potential_label' equals 'below_average'.
# The decision to remove these records is a strategic one to address the class imbalance, which could otherwise bias the model's performance.
df = df.loc[~(df["potential_label"] == "below_average")]



# Creating a Pivot Table for Player Attributes
# In this dataset, we aim to uniquely identify each player in a row and organize all the points given to them.
# We use a pivot table to restructure the data. The 'player_id', 'position_id', and 'potential_label' are set as indices.
# This means each row represents a unique player with their respective position and potential label.
# The 'attribute_value' is used as the value for the pivot table.
# The columns of the pivot table are set to 'attribute_id', representing different attributes scored by scouts.
# This restructuring makes the dataset more suitable for analysis, where each player's attributes are neatly organized in a single row.
table1 = pd.pivot_table(df, values="attribute_value", index=["player_id", "position_id", "potential_label"], columns=["attribute_id"])
table1.head()



# Resetting the index of the pivot table for a cleaner dataframe structure.
table1 = table1.reset_index()
table1.head()

# Converting column names to strings to avoid future issues.
table1.columns = table1.columns.astype(str)




# Re-categorizing variable types in the new dataframe 'table1' after making modifications.
cat_cols, num_cols, cat_but_car = grab_col_names(table1, cat_th=8, car_th=20)

# Displaying and updating the list of numerical columns.
# Excluding the first two columns from the list for further analysis.
num_cols = num_cols[2:]

# Displaying the distribution of values in the 'position_id' column of 'table1'
table1.potential_label.value_counts()

# Observing the class distribution in the potential_label column.
# From this output, we can see there is an imbalance between classes.
"""
             potential_label      Ratio
average                 215  80.068729
highlighted             46  19.931271
##########################################
"""

# Converting the target variable 'potential_label' to a binary format.
labelencoder = LabelEncoder()
table1["potential_label"] = labelencoder.fit_transform(table1["potential_label"])

table1.head()



# Standardize the numerical columns
scaled = StandardScaler().fit_transform(table1[num_cols])
table1[num_cols] = pd.DataFrame(scaled, columns=table1[num_cols].columns)


table1.head()


for col in num_cols:
    print(col, check_outlier(table1, col, 0.05, 0.95))
# False

table1.columns

def base_models(X, y, scoring=["roc_auc", "f1_macro", "accuracy"]):
    print("Base Models....")

    # Create a special SMOTE pipeline
    smote_knn_pipeline = make_pipeline(SMOTE(random_state=42), KNeighborsClassifier())
    smote_gbm_pipeline = make_pipeline(SMOTE(random_state=42), GradientBoostingClassifier())
    smote_adaboost_pipeline = make_pipeline(SMOTE(random_state=42), AdaBoostClassifier())

    class_0_prior = 215 / (215 + 46) # For NaiveBayes
    class_1_prior = 46 / (215 + 46)

    classifiers = [
        ('LR', LogisticRegression(class_weight='balanced')),
        #('KNN', make_pipeline(SMOTE(random_state=42), KNeighborsClassifier())),
        ("SVC", SVC(class_weight='balanced')),
        ("CART", DecisionTreeClassifier(class_weight='balanced')),
        ("RF", RandomForestClassifier(class_weight='balanced')),
        #('Adaboost', make_pipeline(SMOTE(random_state=42), AdaBoostClassifier())),
        #('GBM', make_pipeline(SMOTE(random_state=42), GradientBoostingClassifier())),
        ('XGBoost', XGBClassifier(scale_pos_weight=y.value_counts()[0] / y.value_counts()[1], use_label_encoder=False,
                                  eval_metric='logloss')),
        ('LightGBM', LGBMClassifier(is_unbalance=True, verbose=-1)),
        ('NaiveBayes', GaussianNB(priors=[class_0_prior, class_1_prior])), #I have set the prior parameter to [1, 4] because there are 215 samples in class 0 and 46 samples in class 1.
    ]

    # Loop through each classifier and evaluate it using cross-validation
    for name, classifier in classifiers:
        print(f"Evaluating {name} Classifier:")
        for metric in scoring:
            cv_results = cross_validate(classifier, X, y, cv=3, scoring=metric)
            mean_score = round(cv_results['test_score'].mean(), 4)
            print(f"{metric}: {mean_score} ({name})")


# Explanation:
# - The function tests multiple classifiers like Logistic Regression, SVC, Decision Tree, etc.
# - Class imbalance is addressed by using class weights or algorithms inherently handling imbalance (like LightGBM).
# - The "scoring" parameter allows evaluating models on different metrics like ROC AUC, F1 Macro, and Accuracy.
# - Cross-validation (with 3 folds) is used to evaluate each classifier, providing a mean score for each metric.
# - The function prints the performance of each classifier for each specified metric.



# Let's set up our models

# Define the target variable "y" and the feature matrix "X"
y = table1["potential_label"]
X = table1.drop(["potential_label", "player_id"], axis=1)

base_models(X, y)

"""Base Models....

Evaluating LR Classifier:
roc_auc: 0.813 (LR)
f1_macro: 0.7505 (LR)
accuracy: 0.8193 (LR)

Evaluating SVC Classifier:
roc_auc: 0.8209 (SVC)
f1_macro: 0.7076 (SVC)
accuracy: 0.8007 (SVC)

Evaluating CART Classifier:
roc_auc: 0.708 (CART)
f1_macro: 0.6964 (CART)
accuracy: 0.8047 (CART)

Evaluating RF Classifier:
roc_auc: 0.8886 (RF)
f1_macro: 0.748 (RF)
accuracy: 0.8598 (RF)

Evaluating XGBoost Classifier:
roc_auc: 0.852 (XGBoost)
f1_macro: 0.7666 (XGBoost)
accuracy: 0.8524 (XGBoost)

Evaluating LightGBM Classifier:
roc_auc: 0.8806 (LightGBM)
f1_macro: 0.7385 (LightGBM)
accuracy: 0.8302 (LightGBM)

Evaluating NaiveBayes Classifier:
roc_auc: 0.6623 (NaiveBayes)
f1_macro: 0.4913 (NaiveBayes)
accuracy: 0.5311 (NaiveBayes)"""



# Define hyperparameter grids for Logistic Regression
lr_params = {
    "penalty": ['l1', 'l2', 'elasticnet', 'none'],
    "C": np.logspace(-4, 4, 20),
    "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    "max_iter": list(range(100, 1000, 100))
}

# Define hyperparameter grids for Decision Tree (CART)
cart_params = {
    "max_depth": [None] + list(range(5, 31, 5)),
    "min_samples_split": range(2, 50, 5),
    "min_samples_leaf": range(1, 50, 5),
    "max_features": [None, "auto", "sqrt", "log2"]
}

# Define hyperparameter grids for Random Forest
rf_params = {
    "n_estimators": range(10, 300, 50),
    "max_depth": [None] + list(range(5, 31, 5)),
    "min_samples_split": range(2, 20, 4),
    "min_samples_leaf": range(1, 20, 4),
    "max_features": ['auto', 'sqrt', 'log2']
}

# Define hyperparameter grids for XGBoost
xgboost_params = {
    "n_estimators": range(50, 300, 50),
    "max_depth": range(3, 11, 2),
    "learning_rate": np.arange(0.01, 0.3, 0.05),
    "subsample": np.arange(0.5, 1.05, 0.1),
    "colsample_bytree": np.arange(0.1, 1.05, 0.1)
}

# Define hyperparameter grids for LightGBM
lightgbm_params = {
    "num_leaves": range(20, 200, 30),
    "learning_rate": np.arange(0.01, 0.3, 0.05),
    "n_estimators": range(50, 300, 50),
    "subsample": np.arange(0.5, 1.05, 0.1),
    "colsample_bytree": np.arange(0.5, 1.05, 0.1)
}

# Define hyperparameter grid for Naive Bayes (Gaussian Naive Bayes)
naive_bayes_params = {
    "var_smoothing": np.logspace(-10, -7, 4)
}

# Define hyperparameter grids for Support Vector Classifier (SVC)
svc_params = {
    "C": np.logspace(-3, 2, 6),
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    "degree": [2, 3, 4],
    "gamma": ['scale', 'auto'] + list(np.logspace(-4, 0, 5)),
    "coef0": np.linspace(0, 10, 5)
}



# Create a list of classifiers with their names, classifier objects, and hyperparameter grids


classifiers = [
    ('LR', LogisticRegression(class_weight='balanced', verbose=0), lr_params),
    # ('KNN', make_pipeline(SMOTE(random_state=42), KNeighborsClassifier())),
    ("SVC", SVC(class_weight='balanced', probability=True), svc_params),
    ("CART", DecisionTreeClassifier(class_weight='balanced'), cart_params),
    ("RF", RandomForestClassifier(class_weight='balanced'), rf_params),
    # ('Adaboost', make_pipeline(SMOTE(random_state=42), AdaBoostClassifier())),
    # ('GBM', make_pipeline(SMOTE(random_state=42), GradientBoostingClassifier())),
    ('XGBoost', XGBClassifier(scale_pos_weight=y.value_counts()[0] / y.value_counts()[1], use_label_encoder=False, eval_metric='logloss'), xgboost_params),
    ('LightGBM', LGBMClassifier(is_unbalance=True, verbose=-1), lightgbm_params),
    ('NaiveBayes', GaussianNB(priors=[class_0_prior, class_1_prior]), naive_bayes_params) # I have set the prior parameter to [1, 4] because there are 215 samples in class 0 and 46 samples in class 1.
]

"""I didn't optimize the hyperparameters of models like KNN, Adaboost, and GBM because they cannot handle imbalanced class distribution. 
To address this issue, we should have applied SMOTE, and if we had done so, we would have split the data into training and testing sets 
and then applied SMOTE only to the training set. That's why I didn't include them in this function, but it could have been done that way as well."""

# Calculate class priors for Naive Bayes

class_0_prior = 215 / (215 + 46)
class_1_prior = 46 / (215 + 46)


def hyperparameter_optimization(X, y, classifiers, cv=3, n_iter=100):

    best_models = {}

    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")

        # Calculate ROC AUC and F1 Macro scores before hyperparameter optimization
        roc_auc_before = cross_validate(classifier, X, y, cv=cv, scoring="roc_auc")['test_score'].mean()
        f1_macro_before = cross_validate(classifier, X, y, cv=cv, scoring="f1_macro")['test_score'].mean()
        print(f"ROC AUC (Before): {round(roc_auc_before, 4)}")
        print(f"F1 Macro (Before): {round(f1_macro_before, 4)}")

        # Perform hyperparameter optimization using RandomizedSearchCV
        rs_best = RandomizedSearchCV(classifier, params, cv=cv, n_iter=n_iter, n_jobs=-1, verbose=0, random_state=42).fit(X, y)

        # Set the final model with the best hyperparameters
        final_model = classifier.set_params(**rs_best.best_params_)

        # Calculate ROC AUC and F1 Macro scores after hyperparameter optimization
        roc_auc_after = cross_validate(final_model, X, y, cv=cv, scoring="roc_auc")['test_score'].mean()
        f1_macro_after = cross_validate(final_model, X, y, cv=cv, scoring="f1_macro")['test_score'].mean()
        print(f"ROC AUC (After): {round(roc_auc_after, 4)}")
        print(f"F1 Macro (After): {round(f1_macro_after, 4)}")

        # Check if ROC AUC or F1 Macro improved, and update the best model
        if roc_auc_after > roc_auc_before or f1_macro_after > f1_macro_before:
            print(f"Model improved for {name}, updating with best params.")
            best_models[name] = final_model
        else:
            print(f"Model did not improve for {name}, keeping original.")
            best_models[name] = classifier

        print(f"{name} best params: {rs_best.best_params_}", end="\n\n")

    return best_models

best_models = hyperparameter_optimization(X, y, classifiers, cv=3)


"""Hyperparameter Optimization....
########## LR ##########
ROC AUC (Before): 0.813
F1 Macro (Before): 0.7505
ROC AUC (After): 0.8184
F1 Macro (After): 0.7494
Model improved for LR, updating with best params.
LR best params: {'solver': 'lbfgs', 'penalty': 'l2', 'max_iter': 900, 'C': 0.615848211066026}

########## SVC ##########
ROC AUC (Before): 0.8209
F1 Macro (Before): 0.7076
ROC AUC (After): 0.7857
F1 Macro (After): 0.7838
Model improved for SVC, updating with best params.
SVC best params: {'kernel': 'poly', 'gamma': 'scale', 'degree': 4, 'coef0': 0.0, 'C': 1.0}

########## CART ##########
ROC AUC (Before): 0.7121
F1 Macro (Before): 0.7152
ROC AUC (After): 0.7995
F1 Macro (After): 0.6842
Model improved for CART, updating with best params.
CART best params: {'min_samples_split': 2, 'min_samples_leaf': 6, 'max_features': None, 'max_depth': 5}

########## RF ##########
ROC AUC (Before): 0.8844
F1 Macro (Before): 0.7163
ROC AUC (After): 0.8716
F1 Macro (After): 0.7574
Model improved for RF, updating with best params.
RF best params: {'n_estimators': 60, 'min_samples_split': 6, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 25}

########## XGBoost ##########
ROC AUC (Before): 0.852
F1 Macro (Before): 0.7666
ROC AUC (After): 0.8666
F1 Macro (After): 0.7833
Model improved for XGBoost, updating with best params.
XGBoost best params: {'subsample': 0.5, 'n_estimators': 150, 'max_depth': 7, 'learning_rate': 0.21000000000000002, 'colsample_bytree': 0.6}

########## LightGBM ##########
ROC AUC (Before): 0.8806
F1 Macro (Before): 0.7385
ROC AUC (After): 0.8852
F1 Macro (After): 0.7806
Model improved for LightGBM, updating with best params.
LightGBM best params: {'subsample': 0.7999999999999999, 'num_leaves': 110, 'n_estimators': 50, 'learning_rate': 0.11, 'colsample_bytree': 0.5}

########## NaiveBayes ##########
ROC AUC (Before): 0.6623
F1 Macro (Before): 0.4913
ROC AUC (After): 0.6623
F1 Macro (After): 0.4913
Model did not improve for NaiveBayes, keeping original.
NaiveBayes best params: {'var_smoothing': 1e-10}"""

# Define a function to create a Voting Classifier using the best models and evaluate its performance
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    # Create a Voting Classifier by combining the best models obtained from hyperparameter optimization
    voting_clf = VotingClassifier(estimators=[
        ('Logistic Regression', best_models["LR"]),
        ('SVC', best_models["SVC"]),
        ('RF', best_models["RF"]),
        ('XGBoost', best_models["XGBoost"]),
        ('LightGBM', best_models["LightGBM"]),
        ],
        voting='soft'
    ).fit(X, y)

    # Create a Voting Classifier by combining the best models obtained from hyperparameter optimization
    cv_results = cross_validate(voting_clf, X, y, cv=5, scoring=["accuracy", "f1_macro", "roc_auc"], n_jobs=-1)

    # Print and display the performance metrics of the Voting Classifier
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1 Macro Score: {cv_results['test_f1_macro'].mean()}")
    print(f"ROC AUC: {cv_results['test_roc_auc'].mean()}")

    return voting_clf

voting_clf = voting_classifier(best_models, X, y)

"""Voting Classifier...
Accuracy: 0.8782491582491583
F1 Macro Score: 0.7940223394934771
ROC AUC: 0.8947498238195912"""



# Randomly select a single sample (1 row) from the dataset X using a fixed random seed (random_state=45)
random_user = X.sample(1, random_state=45)

# Use the trained Voting Classifier (voting_clf) to predict the class label for the randomly selected sample
predicted_class = voting_clf.predict(random_user)

# The 'predicted_class' now contains the predicted class label for the random user based on the ensemble of models in the Voting Classifier.

# Save the trained Voting Classifier (voting_clf) to a file using joblib
# This step saves the model to a file on the desktop, allowing you to reuse the model for predictions in the future without retraining it.
joblib.dump(voting_clf, "/Users/mrpurtas/Desktop/voting_clf2.pkl")


# This function is used to visualize the feature importance levels of a given model (voting_clf).
# feature_names: A list or array containing the feature names.
# top_n: Used to visualize the top n most important features, defaults to 10.

def plot_feature_importance(model, feature_names, top_n=10):
    # Get the feature importances of the Random Forest (RF) estimator of the model
    feature_importance = model.named_estimators_['RF'].feature_importances_

    # Convert feature names to an array
    feature_names = np.array(feature_names)

    # Create a DataFrame with the feature names and feature importances
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort by feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Create the visualization
    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'][:top_n], y=fi_df['feature_names'][:top_n])
    plt.title('Feature importance (top 10)')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.show()


# Use the plot_feature_importance function to plot the feature importance levels
# of variables using the voting_clf model
plot_feature_importance(voting_clf, X.columns)


# FINAL COMMENT

# The necessary improvements in scores were achieved by initially trying Grid Search and then, to explore
# a wider range of hyperparameters, using Random Search. However, to get even closer to optimal values, we
# could have further refined the parameter grids to potentially achieve better results.

